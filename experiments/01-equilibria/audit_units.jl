using Test
include(joinpath(@__DIR__,"src/AuditChecks.jl"))
using .AuditChecks
@testset "failure and plateau contract" begin
    @test finite!([1.0,2.0],"valid")==[1.0,2.0]
    for bad in (NaN,Inf,-Inf)
        @test_throws ErrorException finite!(bad,"scalar")
        @test_throws ErrorException finite!([1.0,bad],"vector")
    end
    @test_throws ErrorException required_call(()->nothing,"train";injection="train-exception")
    @test_throws ErrorException required_call(()->nothing,"fd";injection="fd-exception")
    for site in ("fd","final-loss","gradient"), kind in ("nan","inf")
        @test_throws ErrorException finite!(required_call(()->1.0,site;injection="$site-$kind"),site)
    end
    @test plateau([0.015,0.015001,0.015002],0.015)
    @test !plateau([0.01634,0.01998,0.02724],0.01491)
    @test !plateau([0.015,0.015,NaN],0.015)
    @test !plateau([0.015,0.015,Inf],0.015)
    @test !plateau([0.015],0.015)
    @test tolerance(0.01491)<0.000015
end
