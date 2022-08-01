## imports

using Flux
using Flux: Data.DataLoader
using Flux: Losses.logitbinarycrossentropy
using Flux: @epochs
using Metalhead
using CUDA
using Images
using FileIO
using CSV
using DataFrames
using MLUtils
using ProgressMeter
using ProgressLogging
using Pipe
using BSON
using BenchmarkTools
using Plots

## data container implementation

"""
    ImageDataContainer(labels_df, img_dir)

Implements the Data Container interface from MLDataUtils for image datasets.

Implements the functions `nobs` and `getobs`, which are required to use ImageDataContainer
as an argument in a DataLoader for Flux.
"""
struct ImageDataContainer
    labels::AbstractVector
    filenames::AbstractVector{String}
    function ImageDataContainer(labels_df::DataFrame, img_dir::AbstractString)
        filenames = img_dir .* labels_df[!, 1] # first column should be the filenames
        labels = labels_df[!, 2] # second column should be the labels
        return new(labels, filenames)
    end
end

"Gets the number of observations for a given dataset."
function Base.length(dataset::ImageDataContainer)
    return length(dataset.labels)
end

"Gets the i-th to j-th observations (including labels) for a given dataset."
function Base.getindex(dataset::ImageDataContainer, idxs::Union{UnitRange,Vector})
    batch_imgs = map(idx -> load(dataset.filenames[idx]), idxs)
    batch_labels = map(idx -> dataset.labels[idx], idxs)

    "Applies necessary transforms and reshapings to batches and loads them onto GPU to be fed into a model."
    function transform_batch(imgs, labels)
        # convert imgs to 256×256×3×128 array (Height×Width×Color×Number) of floats (values between 0.0 and 1.0) on the gpu
        # arrays need to be sent to gpu inside training loop for garbage collector to work
        batch_X = @pipe hcat(imgs...) |> reshape(_, (HEIGHT, WIDTH, length(labels))) |> channelview |> permutedims(_, (2, 3, 1, 4))
        batch_y = @pipe labels |> reshape(_, (1, length(labels)))

        return (batch_X, batch_y)
    end

    return transform_batch(batch_imgs, batch_labels)
end

## declare constants

const N_CLASSES = 1
const HEIGHT, WIDTH, CHANNELS = 256, 256, 3
const BATCH_SIZE = 64

## create data containers and dataloaders

const dataset_dir = "load_dataset/dataset/"
const train_dir = dataset_dir * "train/"
const test_dir = dataset_dir * "test/"

# dataframes containing filenames for images and corresponding labels
const train_df = DataFrame(CSV.File(dataset_dir * "train_labels.csv"))
const test_df = DataFrame(CSV.File(dataset_dir * "test_labels.csv"))

# ImageDataContainer wrappers for dataframes
# gives interface for getting the actual images and labels as tensors
const train_dataset = ImageDataContainer(train_df, train_dir)
const test_dataset = ImageDataContainer(test_df, test_dir)

# randomly sort train dataset into training and validation sets
const train_set, val_set = splitobs(train_dataset, at=0.7, shuffle=true)

const train_loader = DataLoader(train_set, batchsize=BATCH_SIZE, shuffle=true)
const val_loader = DataLoader(val_set, batchsize=BATCH_SIZE, shuffle=true)
const test_loader = DataLoader(test_dataset, batchsize=BATCH_SIZE)

## training loop

"Stores the history through all the epochs of key training/validation performance metrics."
mutable struct TrainingMetrics
    val_acc::Vector{AbstractFloat}
    val_loss::Vector{AbstractFloat}

    TrainingMetrics(n_epochs::Integer) = new(zeros(n_epochs), zeros(n_epochs))
end

"Trains given model for a given number of epochs and saves the model that performs best on the validation set."
function train(model, n_epochs::Integer, filename::String)
    model = model |> gpu
    optimizer = ADAM()
    params = Flux.params(model[end]) # transfer learning, so only training last layers

    metrics = TrainingMetrics(n_epochs)

    # zero init performance measures for epoch
    epoch_acc = 0.0
    epoch_loss = 0.0

    # so we can automatically save the model with best val accuracy
    best_acc = 0.0

    # X and y are already in the right shape and on the gpu
    # if they weren't, Zygote.jl would throw a fit because it needs to be able to differentiate this function
    loss(X, y) = logitbinarycrossentropy(model(X), y)

    @info "Beginning training loop..."
    for epoch_idx ∈ 1:n_epochs
        @info "Training epoch $(epoch_idx)..."
        # train 1 epoch, record performance
        @withprogress for (batch_idx, (imgs, labels)) ∈ enumerate(train_loader)
            X = @pipe imgs |> gpu |> float32.(_)
            y = @pipe labels |> gpu |> float32.(_)

            gradients = gradient(() -> loss(X, y), params)
            Flux.Optimise.update!(optimizer, params, gradients)

            @logprogress batch_idx / length(enumerate(train_loader))
        end

        # reset variables
        epoch_acc = 0.0
        epoch_loss = 0.0

        @info "Validating epoch $(epoch_idx)..."
        # val 1 epoch, record performance
        @withprogress for (batch_idx, (imgs, labels)) ∈ enumerate(val_loader)
            X = @pipe imgs |> gpu |> float32.(_)
            y = @pipe labels |> gpu |> float32.(_)

            # feed through the model to create prediction
            ŷ = model(X)

            # calculate the loss and accuracy for this batch, add to accumulator for epoch results
            batch_acc = @pipe ((((σ.(ŷ) .> 0.5) .* 1.0) .== y) .* 1.0) |> cpu |> reduce(+, _)
            epoch_acc += batch_acc
            batch_loss = logitbinarycrossentropy(ŷ, y)
            epoch_loss += (batch_loss |> cpu)

            @logprogress batch_idx / length(enumerate(val_loader))
        end
        # add acc and loss to lists
        metrics.val_acc[epoch_idx] = epoch_acc / length(val_set)
        metrics.val_loss[epoch_idx] = epoch_loss / length(val_set)

        # automatically save the model every time it improves in val accuracy
        if metrics.val_acc[epoch_idx] >= best_acc
            @info "New best accuracy: $(metrics.val_acc[epoch_idx])! Saving model out to $(filename).bson"
            BSON.@save joinpath(@__DIR__, "$(filename).bson")
            best_acc = metrics.val_acc[epoch_idx]
        end
    end

    return model, metrics
end

## resnet transfer learning baseline model

# load in saved params from bson
resnet = ResNet(18)
@pipe joinpath(@__DIR__, "resnet18.bson") |> BSON.load(_)[:model] |> Flux.loadmodel!(resnet, _)

# last element of resnet18 is a chain
# since we're removing the last element, we just want to recreate it, but with different number of classes
# probably a more elegant, less hard-coded way to do this, but whatever
baseline_model = Chain(
    resnet.layers[1:end-1],
    Chain(
        AdaptiveMeanPool((1, 1)),
        Flux.flatten,
        Dense(512 => N_CLASSES)
    )
)

baseline_model, baseline_metrics = @time train(baseline_model, 10, "baseline")

plot(baseline_metrics.val_acc, label="resnet18 baseline")
xlabel!("Epoch")
ylabel!("Validation Accuracy")
title!("Validation Accuracy vs Epochs")
savefig("baseline_val_accuracy.png")

## custom twin-network model

# need two dataloaders for each stage, as need two images to feed into the twin network model
const train_loader₁ = DataLoader(train_set, batchsize=BATCH_SIZE, shuffle=true)
const train_loader₂ = DataLoader(train_set, batchsize=BATCH_SIZE, shuffle=true)
const val_loader₁ = DataLoader(val_set, batchsize=BATCH_SIZE, shuffle=true)
const val_loader₂ = DataLoader(val_set, batchsize=BATCH_SIZE, shuffle=true)
const test_loader₁ = DataLoader(test_dataset, batchsize=BATCH_SIZE, shuffle=true)
const test_loader₂ = DataLoader(test_dataset, batchsize=BATCH_SIZE, shuffle=true)

"Custom Flux NN layer which will create twin network from `path` with shared parameters and combine their output with `combine`."
struct Twin{T,F}
    combine::F
    path::T
end

# define the forward pass of the Twin layer
# feeds both inputs, X, through the same path (i.e., shared parameters)
# and combines their outputs
Flux.@functor Twin
(m::Twin)(Xs::Tuple) = m.combine(map(X -> m.path(X), Xs)...)

twin_model = Twin(
    # this layer combines the outputs of the twin CNNs
    Flux.Bilinear((32,32) => 1),
    # this is the architecture that forms the path of the twin network
    Chain(
        # layer 1
        Conv((5,5), 3 => 18, relu),
        MaxPool((3,3), stride=3),
        # layer 2
        Conv((5,5), 18 => 36, relu),
        MaxPool((2,2), stride=2),
        # layer 3
        Conv((3,3), 36 => 72, relu),
        MaxPool((2,2), stride=2),
        Flux.flatten,
        # layer 4
        Dense(19 * 19 * 72 => 64, relu),
        # Dropout(0.1),
        # output layer
        Dense(64 => 32, relu)
    )
)

## training loop

"Trains given twin model for a given number of epochs and saves the model that performs best on the validation set."
function train(model::Twin, n_epochs::Integer, filename::String; is_resnet::Bool=false)
    model = model |> gpu
    optimizer = ADAM()
    params = is_resnet ? Flux.params(model.path[end:end], model.combine) : Flux.params(model) # if custom CNN, need to train all params

    metrics = TrainingMetrics(n_epochs)

    # zero init performance measures for epoch
    epoch_acc = 0.0
    epoch_loss = 0.0

    # so we can automatically save the model with best val accuracy
    best_acc = 0.0

    # X and y are already in the right shape and on the gpu
    # if they weren't, Zygote.jl would throw a fit because it needs to be able to differentiate this function
    loss(Xs, y) = logitbinarycrossentropy(model(Xs), y)

    @info "Beginning training loop..."
    for epoch_idx ∈ 1:n_epochs
        @info "Training epoch $(epoch_idx)..."
        # train 1 epoch, record performance
        @withprogress for (batch_idx, ((imgs₁, labels₁), (imgs₂, labels₂))) ∈ enumerate(zip(train_loader₁, train_loader₂))
            X₁ = @pipe imgs₁ |> gpu |> float32.(_)
            y₁ = @pipe labels₁ |> gpu |> float32.(_)

            X₂ = @pipe imgs₂ |> gpu |> float32.(_)
            y₂ = @pipe labels₂ |> gpu |> float32.(_)

            Xs = (X₁, X₂)
            y = ((y₁ .== y₂) .* 1.0) # y represents if both images have the same label

            gradients = gradient(() -> loss(Xs, y), params)
            Flux.Optimise.update!(optimizer, params, gradients)

            @logprogress batch_idx / length(enumerate(train_loader₁))
        end

        # reset variables
        epoch_acc = 0.0
        epoch_loss = 0.0

        @info "Validating epoch $(epoch_idx)..."
        # val 1 epoch, record performance
        @withprogress for (batch_idx, ((imgs₁, labels₁), (imgs₂, labels₂))) ∈ enumerate(zip(val_loader₁, val_loader₂))
            X₁ = @pipe imgs₁ |> gpu |> float32.(_)
            y₁ = @pipe labels₁ |> gpu |> float32.(_)

            X₂ = @pipe imgs₂ |> gpu |> float32.(_)
            y₂ = @pipe labels₂ |> gpu |> float32.(_)

            Xs = (X₁, X₂)
            y = ((y₁ .== y₂) .* 1.0) # y represents if both images have the same label

            # feed through the model to create prediction
            ŷ = model(Xs)

            # calculate the loss and accuracy for this batch, add to accumulator for epoch results
            batch_acc = @pipe ((((σ.(ŷ) .> 0.5) .* 1.0) .== y) .* 1.0) |> cpu |> reduce(+, _)
            epoch_acc += batch_acc
            batch_loss = logitbinarycrossentropy(ŷ, y)
            epoch_loss += (batch_loss |> cpu)

            @logprogress batch_idx / length(enumerate(val_loader))
        end
        # add acc and loss to lists
        metrics.val_acc[epoch_idx] = epoch_acc / length(val_set)
        metrics.val_loss[epoch_idx] = epoch_loss / length(val_set)

        # automatically save the model every time it improves in val accuracy
        if metrics.val_acc[epoch_idx] >= best_acc
            @info "New best accuracy: $(metrics.val_acc[epoch_idx])! Saving model out to $(filename).bson"
            BSON.@save joinpath(@__DIR__, "$(filename).bson")
            best_acc = metrics.val_acc[epoch_idx]
        end
    end

    return model, metrics
end

twin_model, twin_metrics = @time train(twin_model, 10, "twin_cnn")

plot(twin_metrics.val_acc, label="twin cnn")
xlabel!("Epoch")
ylabel!("Validation Accuracy")
title!("Validation Accuracy vs Epochs")
savefig("twin_cnn_val_accuracy.png")

## twin resnet model

twin_resnet = Twin(
    Flux.Bilinear((32,32) => 1),
    Chain(
        resnet.layers[1:end-1],
        Chain(
            AdaptiveMeanPool((1, 1)),
            Flux.flatten,
            Dense(512 => 32)
        )
    )
)

twin_resnet, twin_resnet_metrics = @time train(twin_resnet, 10, "twin_resnet", is_resnet=true)

plot(twin_resnet_metrics.val_acc, label="twin resnet")
xlabel!("Epoch")
ylabel!("Validation Accuracy")
title!("Validation Accuracy vs Epochs")
savefig("twin_resnet_val_accuracy.png")

## test

function test(model)
    model = model |> gpu

    # zero init performance measures
    avg_acc = 0.0
    avg_loss = 0.0

    @info "Beginning testing..."
    @withprogress for (batch_idx, (imgs, labels)) ∈ enumerate(test_loader)
        X = @pipe imgs |> gpu |> float32.(_)
        y = @pipe labels |> gpu |> float32.(_)

        # feed through the model to create prediction
        ŷ = model(X)

        # calculate the loss and accuracy for this batch, add to accumulator for total results
        batch_acc = @pipe ((((σ.(ŷ) .> 0.5) .* 1.0) .== y) .* 1.0) |> cpu |> reduce(+, _)
        avg_acc += batch_acc
        batch_loss = logitbinarycrossentropy(ŷ, y)
        avg_loss += (batch_loss |> cpu)

        @logprogress batch_idx / length(enumerate(test_loader))
    end
    # add acc and loss to lists
    avg_acc = avg_acc / length(test_dataset)
    avg_loss = avg_loss / length(test_dataset)

    return avg_acc, avg_loss
end

function test(model::Twin)
    model = model |> gpu

    # zero init performance measures
    avg_acc = 0.0
    avg_loss = 0.0

    @info "Beginning testing..."
    @withprogress for (batch_idx, ((imgs₁, labels₁), (imgs₂, labels₂))) ∈ enumerate(zip(test_loader₁, test_loader₂))
        X₁ = @pipe imgs₁ |> gpu |> float32.(_)
        y₁ = @pipe labels₁ |> gpu |> float32.(_)

        X₂ = @pipe imgs₂ |> gpu |> float32.(_)
        y₂ = @pipe labels₂ |> gpu |> float32.(_)

        Xs = (X₁, X₂)
        y = ((y₁ .== y₂) .* 1.0) # y represents if both images have the same label

        # feed through the model to create prediction
        ŷ = model(Xs)

        # calculate the loss and accuracy for this batch, add to accumulator for total results
        batch_acc = @pipe ((((σ.(ŷ) .> 0.5) .* 1.0) .== y) .* 1.0) |> cpu |> reduce(+, _)
        avg_acc += batch_acc
        batch_loss = logitbinarycrossentropy(ŷ, y)
        avg_loss += (batch_loss |> cpu)

        @logprogress batch_idx / length(enumerate(test_loader))
    end
    # add acc and loss to lists
    avg_acc = avg_acc / length(test_dataset)
    avg_loss = avg_loss / length(test_dataset)

    return avg_acc, avg_loss
end

baseline_test_acc, baseline_test_loss = test(baseline_model)
twin_cnn_test_acc, twin_cnn_test_loss = test(twin_model)
twin_resnet_test_acc, twin_resnet_test_loss = test(twin_resnet)

## plot all validation results

plot(baseline_metrics.val_acc, label="resnet baseline")
plot!(twin_metrics.val_acc, label="twin cnn")
plot!(twin_resnet_metrics.val_acc, label="twin resnet")
xlabel!("Epoch")
ylabel!("Validation Accuracy")
title!("Validation Accuracy vs Epochs")
savefig("val_accuracy.png")