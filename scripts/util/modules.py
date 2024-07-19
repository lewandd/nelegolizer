from torch import nn

class Model_n111(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            #nn.Conv1d(60, 32, 8, stride=8),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model_class = {
    "model_n111": Model_n111
}