"""Run with: python -m unittest discover -s reference -p 'test_*.py' -v"""
from __future__ import annotations
import itertools
import unittest
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import ndtr, ndtri
from core import (samples, empirical_score, weighted_score, u_score, cross_score,
    kernel_empirical_score, kernel_cross_score, mmd2_unbiased, rbf, v_triplets,
    fresh_keys, key_weights, effective_sample_size, exact_cfm_field,
    exact_path_samples, gaussian_rbf_expectation, category_teacher,
    conditional_gaussian, matheron_samples)


class ReferenceTests(unittest.TestCase):
    def test_01_invalid_inputs_are_not_silently_repaired(self):
        with self.assertRaises(ValueError): samples([1., np.nan])
        with self.assertRaises(ValueError): u_score([1.], 0.)
        with self.assertRaises(ValueError): key_weights([[0.],[1.]], [0.5], 1.)
        with self.assertRaises(ValueError): exact_cfm_field([0.], 1., [0.,1.], [.5,.5])
        with self.assertRaises(ValueError): cross_score([0.,1.], [1.], 0.)

    def test_02_empirical_crps_binary_equals_brier(self):
        for n1 in range(9):
            x = np.r_[np.ones(n1), np.zeros(8-n1)]
            for y in [0., 1.]:
                self.assertAlmostEqual(empirical_score(x,y), (n1/8-y)**2, places=13)

    def test_03_weighted_cdf_risk_decomposition_exact_enumeration(self):
        p = np.array([.15,.4,.7,.85])
        w = np.array([.1,.2,.3,.4])
        p_target = .6
        lhs = 0.
        for bits in itertools.product([0.,1.],repeat=4):
            b=np.array(bits)
            prob=np.prod(np.where(b == 1,p,1-p))
            lhs += prob*(w@b-p_target)**2
        rhs=(w@p-p_target)**2 + np.sum(w*w*p*(1-p))
        self.assertAlmostEqual(lhs,rhs,places=14)

    def test_04_empirical_covariance_correction_exact_enumeration(self):
        w=np.array([.1,.2,.3,.4])
        mu=np.array([[0.,0.],[1.,0.],[0.,2.],[1.,2.]])
        directions=np.array([[.5,0.],[0.,.7],[.2,.1],[-.1,.4]])
        covs=np.einsum('ni,nj->nij',directions,directions)
        lhs=np.zeros((2,2))
        for signs in itertools.product([-1.,1.],repeat=4):
            y=mu+np.array(signs)[:,None]*directions
            center=w@y
            lhs += np.einsum('n,ni,nj->ij',w,y-center,y-center)/16
        center=w@mu
        rhs=np.einsum('n,nij->ij',w*(1-w),covs)
        rhs+=np.einsum('n,ni,nj->ij',w,mu-center,mu-center)
        np.testing.assert_allclose(lhs,rhs,atol=1e-13)

    def test_05_projection_and_information_price_exact(self):
        q=np.array([.2,.8]); p=np.array([.5,.5])
        def risk(qq,pp): return np.mean((qq-pp)**2+pp*(1-pp))
        v=np.var(q)
        self.assertAlmostEqual(risk(q,p)-risk(np.full(2,q.mean()),p),v)
        p=np.array([.1,.8]); q=np.array([.25,.9])
        v=np.var(q); b=np.var(p); a=np.mean((q-q.mean())*(p-p.mean()))
        for alpha in [0.,.25,.5,1.]:
            qa=(1-alpha)*q+alpha*q.mean()
            gain=risk(q,p)-risk(qa,p)
            self.assertAlmostEqual(gain,alpha*(2-alpha)*v-2*alpha*a,places=13)
        self.assertLessEqual(abs(a),np.sqrt(v*b)+1e-14)
        for alpha in [.25,.5,1.]:
            qa=(1-alpha)*p+alpha*p.mean()
            self.assertAlmostEqual(risk(qa,p)-risk(p,p),alpha**2*b,places=13)

    def test_06_grouped_ensemble_risk_and_u_bias_monte_carlo(self):
        rng=np.random.default_rng(6617)
        n,m_total=60000,16
        truth=rng.integers(0,2,size=n)
        V,W=.25**2,.25-.25**2
        for groups in [1,2,4,16]:
            per_key=m_total//groups
            prob=.5+.25*rng.choice([-1.,1.],size=(n,groups))
            draws=rng.random((n,groups,per_key))<prob[:,:,None]
            p_hat=draws.mean(axis=(1,2))
            emp=(p_hat-truth)**2
            fair=emp-p_hat*(1-p_hat)/(m_total-1)
            expected_emp=.25+V/groups+W/m_total
            expected_fair=.25+(per_key-1)/(m_total-1)*V
            self.assertLess(abs(emp.mean()-expected_emp),6*emp.std()/np.sqrt(n)+2e-4)
            self.assertLess(abs(fair.mean()-expected_fair),6*fair.std()/np.sqrt(n)+2e-4)

    def test_07_cross_score_all_cloned_members_exact(self):
        # Each complete bank contains one Bernoulli(1/2) draw copied 8 times.
        # Q_M is Bernoulli(1/2), despite perfect within-bank dependence.
        cross,fair,emp=0.,0.,0.
        for a,b,y in itertools.product([0.,1.],repeat=3):
            aa=np.full(8,a); bb=np.full(8,b)
            cross+=cross_score(aa,bb,y)/8
            fair+=u_score(aa,y)/8
            emp+=empirical_score(aa,y)/8
        self.assertAlmostEqual(cross,.25)
        self.assertAlmostEqual(fair,.5)
        self.assertAlmostEqual(emp,.5)

    def test_08_fixed_quantile_fair_counterexample_exact_integral(self):
        for m in [2,10,32]:
            x=np.arange(1,m+1)/(m+1)
            # E|x-U|=(x^2+(1-x)^2)/2 for U uniform on [0,1].
            first=np.mean((x*x+(1-x)**2)/2)
            d=np.abs(x[:,None]-x[None,:])
            expected_fair=first-.5*d.sum()/(m*(m-1))
            self.assertAlmostEqual(expected_fair,m/(6*(m+1)),places=14)

    def test_09_same_marginals_same_scores(self):
        rng=np.random.default_rng(91)
        a=rng.normal(size=(16,7)); b=a.copy(); y=rng.normal(size=7)
        for j in range(7): b[:,j]=b[rng.permutation(16),j]
        self.assertAlmostEqual(empirical_score(a,y),empirical_score(b,y),places=13)
        self.assertAlmostEqual(u_score(a,y),u_score(b,y),places=13)
        self.assertGreater(abs(empirical_score(a,y,'energy')-empirical_score(b,y,'energy')),1e-6)

    def test_10_exact_atomic_flow_one_dimensional_quantile_transport(self):
        atoms=np.array([[-1.],[1.5]])
        w=np.array([.35,.65]); n=64; end=.9
        u=(np.arange(n)+.5)/n; z0=ndtri(u)
        def field(t,z): return exact_cfm_field(z,t,atoms,w).ravel()
        sol=solve_ivp(field,(0,end),z0,method='DOP853',rtol=1e-10,atol=1e-11)
        self.assertTrue(sol.success)
        z=sol.y[:,-1]
        cdf=sum(wi*ndtr((z-end*yi[0])/(1-end)) for wi,yi in zip(w,atoms))
        np.testing.assert_allclose(cdf,u,atol=2e-8,rtol=0)

    def test_11_single_atom_field_and_path(self):
        z=np.array([[-2.,.3],[1.,2.]])
        atom=np.array([[.8,-.7]])
        np.testing.assert_allclose(exact_cfm_field(z,.4,atom,[1.]),(atom-z)/.6)
        rng=np.random.default_rng(1)
        np.testing.assert_allclose(exact_path_samples(10,1.,atom,[1.],rng),np.repeat(atom,10,axis=0))

    def test_12_stationary_key_marginals_and_weights(self):
        rng=np.random.default_rng(721)
        keys=rng.normal(size=(40000,8))
        for rho in [0.,.5,.9,1.]:
            kk=fresh_keys(keys,rho,rng)
            np.testing.assert_allclose(kk.mean(axis=0),0,atol=.025)
            np.testing.assert_allclose(kk.var(axis=0),1,atol=.035)
        k=np.array([[-1.,0.],[.2,.1],[2.,3.]])
        np.testing.assert_allclose(key_weights(k,[0.,0.],0),np.ones(3)/3)
        w=key_weights(k,[.2,.1],.999)
        self.assertGreater(w[1],.9999)
        self.assertAlmostEqual(effective_sample_size([.25]*4),4.)

    def test_13_gaussian_kernel_formula_monte_carlo(self):
        rng=np.random.default_rng(13); n=60000
        ma=np.array([.2,-.1]); mb=np.array([-.3,.6])
        ca=np.array([[.3,.05],[.05,.2]]); cb=np.diag([.2,.4])
        a=rng.multivariate_normal(ma,ca,n); b=rng.multivariate_normal(mb,cb,n)
        sq=((a-b)**2).sum(axis=1)
        scores=(np.exp(-sq/(2*.5**2))+np.exp(-sq/2))/2
        exact=gaussian_rbf_expectation(ma,ca,mb,cb,[.5,1.])
        self.assertLess(abs(scores.mean()-exact),6*scores.std()/np.sqrt(n)+1e-4)

    def test_14_teacher_moments_cdf_and_kernel(self):
        rng=np.random.default_rng(14); teacher=category_teacher(7)
        x=teacher.sample(70000,rng); mu,cov=teacher.moments()
        np.testing.assert_allclose(x.mean(axis=0),mu,atol=.016)
        np.testing.assert_allclose(np.cov(x,rowvar=False),cov,atol=.015)
        cdf=teacher.marginal_cdf(mu,0)
        self.assertTrue(((cdf>=0)&(cdf<=1)).all())
        sample=teacher.sample(350,rng)
        discrepancy=teacher.mmd2_from_samples(sample)
        self.assertLess(abs(discrepancy),.02)

    def test_15_label_free_v_triplet_unbiased_discrete_enumeration(self):
        # Q_k are point masses at -1 and +1. No truth appears in the estimator.
        xa=[]; xp=[]; zb=[]
        for a,b in itertools.product([-1.,1.],repeat=2):
            xa.append(a); xp.append(a); zb.append(b)
        estimate=v_triplets(xa,xp,zb,[1.]).mean()
        expected=.5*(1-np.exp(-2))
        self.assertAlmostEqual(estimate,expected,places=14)

    def test_16_linear_gaussian_conditioning_and_matheron(self):
        rng=np.random.default_rng(16)
        mean=np.array([.3,-.1,.5]); cov=np.array([[1.,.2,.1],[.2,.8,.15],[.1,.15,.5]])
        h=np.array([[1.,0.,0.],[0.,.5,1.]])
        obs=np.array([.7,-.2]); noise=np.diag([.1,.2])
        mu,cc,_=conditional_gaussian(mean,cov,h,obs,noise)
        xx=matheron_samples(mean,cov,h,obs,noise,60000,rng)
        np.testing.assert_allclose(xx.mean(axis=0),mu,atol=.018)
        np.testing.assert_allclose(np.cov(xx,rowvar=False),cc,atol=.014)
        # Permutation of observation rows is exactly information-preserving.
        p=np.array([[0.,1.],[1.,0.]])
        mu2,cc2,_=conditional_gaussian(mean,cov,p@h,p@obs,p@noise@p.T)
        np.testing.assert_allclose(mu,mu2,atol=1e-13)
        np.testing.assert_allclose(cc,cc2,atol=1e-13)

    def test_17_kernel_cross_score_consistency_point_forecast(self):
        x=np.repeat([[.2,-.4]],8,axis=0); y=np.array([.3,.6])
        self.assertAlmostEqual(kernel_cross_score(x,x,y),kernel_empirical_score(x,y),places=13)
        self.assertAlmostEqual(mmd2_unbiased(x,x),0,places=13)


if __name__=='__main__': unittest.main(verbosity=2)
