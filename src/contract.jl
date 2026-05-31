#


function contraction_output(
    tensor1::BlockSparseTensor,
    labelstensor1,
    tensor2::BlockSparseTensor,
    labelstensor2,
    labelsR
)
    indsR =
        ITensors.NDTensors.contract_inds(inds(tensor1), labelstensor1, inds(tensor2), labelstensor2, labelsR)
    # TensorR =NDTensors.contraction_output_type(typeof(tensor1), typeof(tensor2), indsR)
    blockoffsetsR, contraction_plan = ITensors.NDTensors.contract_blockoffsets(
        blockoffsets(tensor1),
        inds(tensor1),
        labelstensor1,
        blockoffsets(tensor2),
        inds(tensor2),
        labelstensor2,
        indsR,
        labelsR
    )
    return blockoffsetsR, contraction_plan, indsR
end
function ITensors.NDTensors.contract!(
    R::BlockSparseTensor,
    labelsR,
    tensor1::BlockSparseTensor,
    labelstensor1,
    tensor2::BlockSparseTensor,
    labelstensor2,
    a::Number,
    b::Number
)
    off_R = ITensors.NDTensors.blockoffsets(R)
    #generate contract plan
    off_R1, contraction_plan, indsR = contraction_output(
        tensor1, labelstensor1, tensor2, labelstensor2, labelsR
    )
    check = off_R == off_R1
    if !check
        # error("The contraction result doesn't match the input tensor: off_R = $off_R, off_R1 = $off_R1")
        R = reshape(R, off_R1, indsR)
    end
    R = ITensors.NDTensors.contract!(R, labelsR, tensor1, labelstensor1, tensor2, labelstensor2, contraction_plan)
    return R
end
function conTract!(C::ITensor, A::ITensor, B::ITensor)
    labelsR, labelsA, labelsB = ITensors.compute_contraction_labels(inds(C), inds(A), inds(B))
    tensor1 = tensor(A)
    tensor2 = tensor(B)
    R = tensor(C)
    off_R = ITensors.NDTensors.blockoffsets(R)
    off_R1, contraction_plan, indsR = contraction_output(
        tensor1, labelsA, tensor2, labelsB, labelsR
    )

    check = off_R == off_R1
    if !check
        R = reshape(R, off_R1, indsR)
    end
    R = ITensors.NDTensors.contract!(R, labelsR, tensor1, labelsA, tensor2, labelsB, contraction_plan)
    ITensors.setstorage!(C, ITensors.storage(R))
    return C
end
