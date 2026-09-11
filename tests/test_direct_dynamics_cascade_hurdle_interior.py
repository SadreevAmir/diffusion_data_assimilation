import torch

from assim_lib.direct_dynamics_cascade_hurdle_interior import apply_hurdle_interior


def test_sic_hurdle_has_zero_atom_and_strict_interior():
    values=torch.tensor([-1.0,0.0,0.5,1.0])
    result=apply_hurdle_interior(values,1.0,"sic",0.0,1.0,0.0)
    assert torch.equal(result[:2],torch.zeros(2,dtype=torch.float64))
    assert torch.all((result[2:]>0)&(result[2:]<1))


def test_sit_identity_is_recovered_at_zero_threshold():
    values=torch.tensor([-1.0,0.0,0.5,2.0])
    result=apply_hurdle_interior(values,0.4,"sit",0.0,1.0,0.0)
    torch.testing.assert_close(result,torch.tensor([0.0,0.0,0.5,2.0],dtype=torch.float64))


def test_transform_is_memberwise_and_monotone():
    member=torch.tensor([0.1,0.3,0.8])
    alone=apply_hurdle_interior(member,0.4,"sic",0.1,1.2,-0.2)
    together=apply_hurdle_interior(torch.stack((member,member+0.01)),0.4,"sic",0.1,1.2,-0.2)
    torch.testing.assert_close(alone,together[0]); assert torch.all(alone[1:]>=alone[:-1])
