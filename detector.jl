## imports

using Flux
using Flux: Data.DataLoader
using Flux: Losses.logitcrossentropy
using Flux: @epochs
using Metalhead
using Images
using FileIO
using CSV
using DataFrames
using MLUtils
using ProgressMeter
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
function Base.getindex(dataset::ImageDataContainer, idxs::UnitRange)
    batch_imgs = map(idx -> load(dataset.filenames[idx]), idxs)
    batch_labels = map(idx -> dataset.labels[idx], idxs)

    "Applies necessary transforms and reshapings to batches and loads them onto GPU to be fed into a model."
    function transform_batch(imgs, labels)
        # convert imgs to 256×256×3×128 array (Height×Width×Color×Number) of floats (values between 0.0 and 1.0) on the gpu
        # arrays are sent to gpu at optimal time based on benchmark performance
        batch_X = @pipe hcat(imgs...) |> reshape(_, (HEIGHT, WIDTH, length(labels))) |> gpu |> channelview |> permutedims(_, (2, 3, 1, 4)) |> float32.(_)
        batch_y = @pipe labels |> reshape(_, (1, length(labels))) |> gpu |> float32.(_)

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

train_df = DataFrame(CSV.File(dataset_dir * "train_labels.csv"))
test_df = DataFrame(CSV.File(dataset_dir * "test_labels.csv"))

train_dataset = ImageDataContainer(train_df, train_dir)
test_dataset = ImageDataContainer(test_df, test_dir)

train_loader = DataLoader(train_dataset, batchsize=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batchsize=BATCH_SIZE)

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
) |> gpu

## training loop

for (X, y) ∈ test_loader
    # @time X, y = transform_batch(imgs, labels)
    println(size(X), size(y))
    println(typeof(X), typeof(y))
end

function train(model, n_epochs)
    model = model
    optimizer = ADAM()
    params = Flux.params(model)

    # X and y are already in the right shape and on the gpu
    # if they weren't, Zygote.jl would throw a fit because it needs to be able to differentiate this function
    function loss(X, y)
        ŷ = model(X)
        return logitcrossentropy(ŷ, y)
    end

    @epochs n_epochs Flux.train!(loss, params, train_loader, optimizer)
    return model
end

train(baseline_model, 2)
