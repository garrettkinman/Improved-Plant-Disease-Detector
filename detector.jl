## imports

using Flux
using Flux: Data.DataLoader
using Flux: Losses.logitbinarycrossentropy
using Flux: @epochs
using Metalhead
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

# "Gets the i-th observation (including label) for a given dataset."
# function Base.getindex(dataset::ImageDataContainer, idx::Integer)
#     if idx > length(dataset)
#         throw(ArgumentError("The dataset is not that large!"))
#     else
#         img = load(dataset.filenames[idx])
#         label = dataset.labels[idx]

#         return (img, label)
#     end
# end

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

N_CLASSES = 1
HEIGHT, WIDTH, CHANNELS = 256, 256, 3
BATCH_SIZE = 128

## create data containers and dataloaders

dataset_dir = "load_dataset/dataset/"
train_dir = dataset_dir * "train/"
test_dir = dataset_dir * "test/"

# dataframes containing filenames for images and corresponding labels
train_df = DataFrame(CSV.File(dataset_dir * "train_labels.csv"))
test_df = DataFrame(CSV.File(dataset_dir * "test_labels.csv"))

# ImageDataContainer wrappers for dataframes
# gives interface for getting the actual images and labels as tensors
train_dataset = ImageDataContainer(train_df, train_dir)
test_dataset = ImageDataContainer(test_df, test_dir)

# randomly sort train dataset into training and validation sets
train_set, val_set = splitobs(train_dataset, at=0.7, shuffle=true)

train_loader = DataLoader(train_set, batchsize=BATCH_SIZE, shuffle=true)
val_loader = DataLoader(val_set, batchsize=BATCH_SIZE, shuffle=true)
test_loader = DataLoader(test_dataset, batchsize=BATCH_SIZE)

## training loop

"Stores the history through all the epochs of key training/validation performance metrics."
mutable struct TrainingMetrics
    train_acc::Vector{AbstractFloat}
    train_loss::Vector{AbstractFloat}
    val_acc::Vector{AbstractFloat}
    val_loss::Vector{AbstractFloat}

    TrainingMetrics(n_epochs::Integer) = new(zeros(n_epochs), zeros(n_epochs), zeros(n_epochs), zeros(n_epochs))
end

"Trains given model for a given number of epochs and saves the model that performs best on the validation set."
function train(model, n_epochs::Integer)
    model = model |> gpu
    optimizer = ADAM()
    params = Flux.params(model)

    metrics = TrainingMetrics(n_epochs)

    # zero init performance measures for epoch
    acc = 0.0
    loss = 0.0

    # so we can automatically save the model with best val accuracy
    best_acc = 0.0

    # X and y are already in the right shape and on the gpu
    # if they weren't, Zygote.jl would throw a fit because it needs to be able to differentiate this function
    function loss_func(X, y)
        ŷ = model(X)
        batch_loss = logitbinarycrossentropy(ŷ, y)

        # this is not ideal design, but I'm not sure how else to calculate training accuracies
        # besides using this loss function
        batch_acc = @pipe ((((σ.(ŷ) .> 0.5) .* 1.0) .== y) .* 1.0) |> cpu |> reduce(+, _)
        acc += batch_acc
        loss += (batch_loss |> cpu)

        return batch_loss
    end

    @info "Beginning training loop..."
    @showprogress for epoch_idx ∈ 1:n_epochs
        @info "Training epoch $(epoch_idx)..."
        # train 1 epoch, record performance
        @withprogress for (batch_idx, (imgs, labels)) ∈ enumerate(train_loader)
            X = @pipe imgs |> gpu |> float32.(_)
            y = @pipe labels |> gpu |> float32.(_)

            gradients = gradient(() -> loss_func(X, y), params)
            Flux.Optimise.update!(optimizer, params, gradients)

            @logprogress batch_idx / length(enumerate(train_loader))
        end
        # Flux.train!(loss_func, params, train_loader, optimizer)
        metrics.train_acc[epoch_idx] = acc / length(train_set)
        metrics.train_loss[epoch_idx] = loss / length(train_set)

        # reset variables
        acc = 0.0
        loss = 0.0

        @info "Validating epoch $(epoch_idx)..."
        # val 1 epoch, record performance
        @withprogress for (batch_idx, (imgs, labels)) ∈ enumerate(val_loader)
            X = @pipe imgs |> gpu |> float32.(_)
            y = @pipe labels |> gpu |> float32.(_)
            batch_loss = loss_func(X, y)

            @logprogress batch_idx / length(enumerate(val_loader))
        end
        # add acc and loss to lists
        metrics.val_acc[epoch_idx] = acc / length(val_set)
        metrics.val_loss[epoch_idx] = loss / length(val_set)

        # automatically save the model every time it improves in val accuracy
        if metrics.val_acc[epoch_idx] >= best_acc
            @info "New best accuracy: $(metrics.val_acc[epoch_idx])! Saving model out to baseline.bson"
            BSON.@save joinpath(@__DIR__, "baseline.bson")
            best_acc = metrics.val_acc[epoch_idx]
        end

        # reset variables
        acc = 0.0
        loss = 0.0
    end

    # @epochs n_epochs Flux.train!(loss, params, train_loader, optimizer)
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
        MLUtils.flatten,
        Dense(512 => N_CLASSES)
    )
)

@time train(baseline_model, 10)