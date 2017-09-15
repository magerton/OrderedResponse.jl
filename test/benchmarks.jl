β = [0.2, -1.0]
γ = [-0.4, 0.5]

θtrue = [β..., γ...]

function θstata(model::Symbol)
    model == :probit && return [.20154344,  -.95584277,  -.36636477,    .4843079]
    model == :logit  && return [.33638928, -1.6251676, -.62388885, .82423386]
    throw(error())
end

function vcovstata(model::Symbol)

    model == :probit && return [ .00159179  -.00028434   -.0000851   .00003368;
                                -.00028434   .00252983   .00032415  -.00026669;
                                 -.0000851   .00032415   .00210196   .00095648;
                                 .00003368  -.00026669   .00095648   .00218769]
    model == :logit && return [.00470505  -.00105662  -.00032196   .00017456;
                              -.00105662   .00868674   .00130306  -.00135786;
                              -.00032196   .00130306   .00606232   .00258262;
                               .00017456  -.00135786   .00258262   .00647028]
    throw(error())
end

function θpolr(model::Symbol)
    model == :logit  && return [0x1.58766ddc938dfp-2, -0x1.a00afbb34ab94p+0, -0x1.3f6e5c3a0abc6p-1, 0x1.a601fa3d8b336p-1]
    model == :probit && return [0x1.9cc2cf36aaba6p-3, -0x1.e9643962405bap-1, -0x1.772853f0930d9p-2, 0x1.efee6817f6849p-2 ]
    throw(error())
end

function vcovpolr(model::Symbol)
    model == :logit  && return reshape([0x1.3459b8009a617p-8, -0x1.14fcb4689d42cp-10, -0x1.5199a6ddeb182p-12, 0x1.6e13a5ae9d64cp-13, -0x1.14fcb4689d428p-10, 0x1.1ca59d7d22ea6p-7, 0x1.55971fae05968p-10, -0x1.63f420acf772ap-10, -0x1.5199a6ddeb18cp-12, 0x1.55971fae0598p-10, 0x1.8d4cd2d9c3a5cp-8, 0x1.52825fc23e733p-9, 0x1.6e13a5ae9d678p-13, -0x1.63f420acf771cp-10, 0x1.52825fc23e738p-9, 0x1.a8094dae89bfep-8], 4, 4)
    model == :probit && return reshape([0x1.a1478d41c4cc5p-10, -0x1.2a27562a3841ap-12, -0x1.64f3df22abaacp-14, 0x1.1a8ad6e51a082p-15, -0x1.2a27562a3841p-12, 0x1.4b970d34ca5eap-9, 0x1.53e4445a38accp-12, -0x1.17a49b5a7ce5cp-12, -0x1.64f3df22abaafp-14, 0x1.53e4445a38ad6p-12, 0x1.1381eb141b246p-9, 0x1.f579486d2078p-11, 0x1.1a8ad6e51a068p-15, -0x1.17a49b5a7ce5ep-12, 0x1.f579486d2078ep-11, 0x1.1ebe907559363p-9], 4, 4)
    throw(error())
end
