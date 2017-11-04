require 'nn'

function get_model()
    generator = nn.Sequential()
    generator:add(nn.Linear(100, 1200))
    generator:add(nn.ReLU())
    generator:add(nn.Linear(1200, 1200))
    generator:add(nn.ReLU())
    generator:add(nn.Linear(1200, 32*32))
    generator:add(nn.Tanh())
    generator:add(nn.Reshape(32, 32))


    discriminator = nn.Sequential()
    discriminator:add(nn.Reshape(1, 32*32))
    discriminator:add(nn.Linear(32*32, 240))
    discriminator:add(nn.ReLU())
    discriminator:add(nn.Linear(240, 240))
    discriminator:add(nn.ReLU())
    discriminator:add(nn.Linear(240, 2))
    discriminator:add(nn.LogSoftMax())

    return generator, discriminator
end
