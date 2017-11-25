require 'nn'
function init(model, mean, std) 
    queue = {model}
    local idx = 1
    while idx <= #queue do
        local t = torch.type(queue[idx])
        --　print("proprocessing ", t)
        if t == "nn.Sequential" or t == "nn.ParallelTable" then
            for m_idx = 1, #queue[idx].modules do
                local m = torch.type(queue[idx].modules[m_idx])
                -- print("insert modules", m)
                table.insert(queue, queue[idx].modules[m_idx])
            end
        elseif t == "nn.Linear" then
            -- print("init weight, bias", t)
--             print("before")
--             print(queue[idx].weight:mean())
--             print(queue[idx].weight:std())
            queue[idx].weight:add(-queue[idx].weight:mean()+mean)
            queue[idx].weight:div(queue[idx].weight:std()):mul(std)
--             print("after")
--             print(queue[idx].weight:mean())
--             print(queue[idx].weight:std())
            queue[idx].bias:zero()
        else
            -- print("ignore module", t)
        end
        idx = idx + 1
    end

end

function get_model()
    generator = nn.Sequential()
    local p = nn.ParallelTable()

    local noiseBranch = nn.Sequential()
    noiseBranch:add(nn.Linear(100, 256))
    noiseBranch:add(nn.BatchNormalization(256))
    noiseBranch:add(nn.ReLU())

    local classBranch = nn.Sequential()
    classBranch:add(nn.Linear(10, 256))
    classBranch:add(nn.BatchNormalization(256))
    classBranch:add(nn.ReLU())

    p:add(noiseBranch)
    p:add(classBranch)

    generator:add(p)
    generator:add(nn.JoinTable(2))
    generator:add(nn.Linear(512, 512))
    generator:add(nn.BatchNormalization(512))
    generator:add(nn.ReLU())
    generator:add(nn.Linear(512, 1024))
    generator:add(nn.BatchNormalization(1024))
    generator:add(nn.ReLU())
    generator:add(nn.Linear(1024, 32*32))
    generator:add(nn.Tanh())
    generator:add(nn.Reshape(32, 32))



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
