require 'nn';
local QLinear, parent = torch.class('nn.QLinear', 'nn.Linear')
-- override the contructor to have additional range of initialization
function QLinear:__init(inputSize, outputSize, mean, std)
    parent:__init(inputSize, outputSize)
    -- self:reset(mean, std)
end

-- override the reset method to use custom weight initialization
function QLinear:reset(mean, std)
    -- self.weights:normal(mean, std)
    -- self.bias:normal(mean, std)
end
