require 'nn'

function get_model()
    generator = nn.Sequential()
    local p = nn.ParallelTable()

    local noiseBranch = nn.Sequential()
    noiseBranch:add(nn.Reshape(100, 1, 1, true))
    noiseBranch:add(nn.SpatialFullConvolution(100, 128, 4, 4))
    noiseBranch:add(nn.ReLU())

    local classBranch = nn.Sequential()
    classBranch:add(nn.Reshape(10, 1, 1, true))
    classBranch:add(nn.SpatialFullConvolution(10, 128, 4, 4))
    classBranch:add(nn.ReLU())

    p:add(noiseBranch)
    p:add(classBranch)

    generator:add(p)
    generator:add(nn.JoinTable(2))
    generator:add(nn.SpatialFullConvolution(256, 128, 4, 4, 2, 2, 1, 1))
    generator:add(nn.ReLU())
    
    generator:add(nn.SpatialFullConvolution(128, 128, 4, 4, 2, 2, 1, 1))
    generator:add(nn.ReLU())
    
    
    generator:add(nn.SpatialFullConvolution(128, 1, 4, 4, 2, 2, 1, 1))
    generator:add(nn.Tanh())
    generator:add(nn.Reshape(32, 32, true))
    
    -- generator:add(nn.Linear(512, 512))
    -- generator:add(nn.BatchNormalization(512))
    -- generator:add(nn.ReLU())
    -- generator:add(nn.Linear(512, 1024))
    -- generator:add(nn.BatchNormalization(1024))
    -- generator:add(nn.ReLU())
    -- generator:add(nn.Linear(1024, 32*32))

    



    discriminator = nn.Sequential()

    local p = nn.ParallelTable()

    local imgBranch = nn.Sequential()
    imgBranch:add(nn.Reshape(32*32))


    imgBranch:add(nn.Linear(32*32, 1024))
    imgBranch:add(nn.LeakyReLU(0.2))

    local classBranch = nn.Sequential()
    classBranch:add(nn.Linear(10, 1024))
    imgBranch:add(nn.LeakyReLU(0.2))

    p:add(imgBranch)
    p:add(classBranch)

    discriminator:add(p)
    discriminator:add(nn.JoinTable(2))
    discriminator:add(nn.Linear(2048, 512))
    discriminator:add(nn.BatchNormalization(512))
    discriminator:add(nn.LeakyReLU(0.2))
    
    discriminator:add(nn.Linear(512, 256))
    discriminator:add(nn.BatchNormalization(256))
    discriminator:add(nn.LeakyReLU(0.2))
    
    discriminator:add(nn.Linear(256, 2))
    discriminator:add(nn.LogSoftMax())
    
    return generator, discriminator
end
