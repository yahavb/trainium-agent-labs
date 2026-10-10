"""One controlled source mutation per independent clean pattern."""

def inject(source,spec):
    kind=spec['kind']
    rules={
      'copy':('tile=nl.ndarray((2,3)','tile=nl.ndarray((2,4)','DMA_SHAPE_MISMATCH','matching_dma_extents'),
      'slice':('tile=nl.ndarray((2,3)','tile=nl.ndarray((6,)','INVALID_TENSOR_DIMENSIONS','explicit_onchip_rank'),
      'multiply':('nisa.tensor_scalar','nisa.scalar_mul','INVALID_API_FUNCTION','supported_scalar_api'),
      'add':('operand0=.25','operand=.25','INVALID_API_ARGUMENT','valid_scalar_keyword'),
      'binary':('nisa.tensor_tensor','nisa.magic_multiply','INVALID_API_FUNCTION','supported_binary_api'),
      'sum':('keepdims=True','keepdims=False','INVALID_TENSOR_DIMENSIONS','reduction_rank'),
      'max':('keepdims=True','keepdims=False','INVALID_TENSOR_DIMENSIONS','reduction_rank'),
      'matmul':('psum=nl.ndarray((2,3)','psum=nl.ndarray((1,2)','INVALID_TENSOR_DIMENSIONS','matmul_destination'),
      'accum':('accumulate=(i>0)','accumulate=False','NUMERICAL_MISMATCH','contraction_accumulation'),
      'partition':('tile=nl.ndarray((128,3)','tile=nl.ndarray((256,3)','INVALID_TENSOR_DIMENSIONS','partition_limit'),
      'columns':('for i in range(2):','for i in range(1):','NUMERICAL_MISMATCH','output_coverage'),
      'psum':('buffer=nl.psum','buffer=nl.shared_hbm','INVALID_BUFFER_PLACEMENT','onchip_copy_regions'),
    }
    old,new,category,cause=rules[kind]
    if source.count(old)!=1:raise ValueError('Mutation target must be unique')
    return source.replace(old,new,1),category,cause
