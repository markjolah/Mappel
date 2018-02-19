# test_mappel_1DFixedSigma.py
# Mark J. Olah (mjo\@cs.unm DOT edu)
# 2018
# Mappel 1D fixed sigma model tests
        
def test_psf_sigma_1D(model1DFixedSigma):
    """Check psf_sigma get and set properties."""
    model = model1DFixedSigma
    assert model.psf_sigma >= model.global_min_psf_sigma
    psf_sigma = model.psf_sigma
    model.psf_sigma = psf_sigma+0.2
    assert psf_sigma+0.2 == model.psf_sigma
