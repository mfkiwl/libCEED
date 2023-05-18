#include <torch/torch.h>
#include <iostream>

extern "C" int create_tensor();

int create_tensor() {

  // Create tensor on CPU
  torch::Tensor atensor = torch::rand({64, 64});
  //std::cout << "Tensor created on device " << atensor.device().type() << std::endl;
  //std::cout << "Tensor is : " << tensor << "\n" << std::endl;
  torch::Tensor btensor = torch::rand({64, 64});
  torch::Tensor ctensor = torch::matmul(atensor,btensor);
  std::cout << "Performed Torch matrix multiply" << std::endl;

  return 0;
}
