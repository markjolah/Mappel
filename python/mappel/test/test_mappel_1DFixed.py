# test_mappel.py
# Mark J. Olah (mjo\@cs.unm DOT edu)
# 2018
# Mappel 1D fixed sigma model tests
        
def test_size_1DFixed(model1DFixed):
    """Check psf_sigma get and set properties."""
    assert(model1DFixed.psf_sigma > 0)
    psf_sigma = model1DFixed.psf_sigma
    model1DFixed.psf_sigma = psf_sigma+0.2
    assert(psf_sigma+0.2 == model1DFixed.psf_sigma)
