require 'nn'

function get_model()
    generator = nn.Sequential()
    local p = nn.ParallelTable()

    local noiseBranch = nn.Sequential()
    noiseBranch:add(nn.Linear(100, 200))
    noiseBranch:add(nn.ReLU())

    local classBranch = nn.Sequential()
    classBranch:add(nn.Linear(10, 1000))
    classBranch:add(nn.ReLU())

    p:add(noiseBranch)
    p:add(classBranch)

    generator:add(p)
    generator:add(nn.JoinTable(1))

    generator:add(nn.Linear(1200, 32*32))
    generator:add(nn.Tanh())
    generator:add(nn.Reshape(32, 32))



   discriminator = nn.Sequential()


    local p = nn.ParallelTable()

    local imgBranch = nn.Sequential()
    imgBranch:add(nn.Reshape(32*32))


    imgBranch:add(nn.Maxout(32*32, 240, 5))
    imgBranch:add(nn.ReLU())
    imgBranch:add(nn.Maxout(240, 240, 4))
    imgBranch:add(nn.ReLU())

    local classBranch = nn.Sequential()
    classBranch:add(nn.Maxout(10, 50, 5))
    classBranch:add(nn.ReLU())
    classBranch:add(nn.Maxout(50, 240, 4))
    classBranch:add(nn.ReLU())

    p:add(imgBranch)
    p:add(classBranch)

    discriminator:add(p)
    discriminator:add(nn.JoinTable(1))
    discriminator:add(nn.Linear(480, 2))
    discriminator:add(nn.LogSoftMax())
    return generator, discriminator
end
