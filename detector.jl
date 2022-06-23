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

## create data containers and dataloaders

dataset_dir = "load_dataset/dataset/"
train_dir = dataset_dir * "train/"
test_dir = dataset_dir * "test/"

train_df = DataFrame(CSV.File(dataset_dir * "train_labels.csv"))
test_df = DataFrame(CSV.File(dataset_dir * "test_labels.csv"))

train_dataset = ImageDataContainer(train_df, train_dir)
test_dataset = ImageDataContainer(test_df, test_dir)

train_loader = DataLoader(train_dataset, batchsize=128)
test_loader = DataLoader(test_dataset, batchsize=128)

@time for (batch_X, batch_y) ∈ test_loader
    println(typeof(batch_X))
end

typeof(1:3)

batch_X, batch_y = getindex(train_dataset, 1:2)

typeof(batch_X)

# this converts 256×256 matrix of RGB pixels into 256×256×3 array
@pipe batch_X[1] |> channelview |> permutedims(_, (2, 3, 1))

Threads.nthreads()

# TODO: a batch should be a 256×256×3×128 array