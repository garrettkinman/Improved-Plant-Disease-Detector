## imports

using Flux
using Flux: Data.DataLoader
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

"Gets the i-th observation (including label) for a given dataset."
function Base.getindex(dataset::ImageDataContainer, idx::Integer)
    if idx > length(dataset)
        throw(ArgumentError("The dataset is not that large!"))
    else
        img = load(dataset.filenames[idx])
        label = dataset.labels[idx]

        return (img, label)
    end
end

"Gets the i-th to j-th observations (including labels) for a given dataset."
function Base.getindex(dataset::ImageDataContainer, idxs::UnitRange)
    batch_imgs = map(idx -> load(dataset.filenames[idx]), idxs)
    batch_labels = map(idx -> dataset.labels[idx], idxs)

    # for i ∈ idx
    #     img, label = getindex(dataset, i)
    #     push!(batch_imgs, img)
    #     push!(batch_labels, label)
    # end
    return (batch_imgs, batch_labels)
end

## declare constants

N_CLASSES = 2
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

@time for (imgs, labels) ∈ test_loader
    # convert to 256×256×3×128 array (Height×Width×Color×Number) of floats (values between 0.0 and 1.0) on the GPU
    # arrays are sent to gpu at optimal time based on benchmark performance
    batch_X = @pipe hcat(imgs...) |> reshape(_, (HEIGHT, WIDTH, length(labels))) |> gpu |> channelview |> permutedims(_, (2, 3, 1, 4)) |> float32.(_)
    batch_y = @pipe labels |> gpu |> float32.(_)
end

## resnet transfer learning baseline model

# load in saved params from bson
resnet = ResNet(18)
@pipe joinpath(@__DIR__, "resnet18.bson") |> BSON.load(_)[:model] |> Flux.loadmodel!(resnet, _)

resnet.layers

resnet.layers[end][end]

baseline_model = Chain(
    resnet.layers[1:end-1],
    Chain(
        AdaptiveMeanPool((1, 1)),
        MLUtils.flatten,
        Dense(512 => 2)
    )
)

batch_X, batch_y = getindex(train_dataset, 1:2)
float32.(batch_y)

batch = @pipe hcat(batch_X...) |> reshape(_, (256, 256, length(batch_y))) |> channelview |> permutedims(_, (2, 3, 1, 4)) |> float32.(_)
gpu(batch)
typeof(batch)
batch[:,:,:,1]
colorview(RGB, permutedims(batch[:, :, :, 1], (3, 1, 2)))