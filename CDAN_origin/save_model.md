```python
import torch
model = torch.load('snapshot/tiered_CDANE/iter_90000_model.pth.tar')
base_network= list(model.children())[0].module
state_dict = base_network.cpu().state_dict()
state_dict = {k: v for k, v in state_dict.items() if 'feature_layers' not in k and 'bottleneck' not in k and 'fc' not in k and 'num_batches_tracked' not in k}
torch.save(state_dict, 'tiered_cpu.pth')
```
