# tessperiment

Experiments with [TESS alerts](https://archive.stsci.edu/prepds/tess-data-alerts/) by [Brett Morris (UW)](http://brettmorr.is). Comments/questions? Tweet [@brettmor](https://twitter.com/brettmor).

##### TIC 52368076 [b](https://archive.stsci.edu/hlsps/tess-data-alerts/hlsp_tess-data-alerts_tess_phot_00052368076-s01s02-01_tess_v1_dvs.pdf) and [c](https://archive.stsci.edu/hlsps/tess-data-alerts/hlsp_tess-data-alerts_tess_phot_00052368076-s01s02-02_tess_v1_dvs.pdf)

Multiplanet system with 2:1 period ratio. I fit TIC 52368076 for transit timing variations, and find that the innermost planet has at least one significant departure from a linear ephemeris, but there are insufficient measurements to constrain the masses of the planets via [TTVFast](http://adsabs.harvard.edu/abs/2014ApJ...787..132D). Unfortunately, according to [tess-point](https://github.com/christopherburke/tess-point) this target will only be observed in Sectors 1 & 2, so this is all we've got.

##### TIC 260271203 [B/b](https://archive.stsci.edu/hlsps/tess-data-alerts/hlsp_tess-data-alerts_tess_phot_00260271203-s01s02-01_tess_v1_dvs.pdf)

This object appears to be orbiting a star with the following properties: Rstar: 1.21, Teff: 5963.0 K, Logg: 4.31, and appears to have a *very* deep transit depth. Here I test whether or not the large radius can be explained by unocculted starspots in the vein of [Morris et al 2018](http://adsabs.harvard.edu/abs/2018AJ....156...91M) and find that the most likely radius of the planet from ingress/egress duration is still quite large. According to [tess-point](https://github.com/christopherburke/tess-point) this target will be observed in Sectors 1-13, so there's *lots* more data coming for this target. 

##### TIC 280830734 [b](https://archive.stsci.edu/hlsps/tess-data-alerts/hlsp_tess-data-alerts_tess_phot_00280830734-s01s02-01_tess_v1_dvs.pdf)

Grazing transit of a variable, slightly evolved star with Rstar: 1.61, Teff: 6340.3 K, Logg: 4.18. Here I test whether or not there's evidence for unocculted starspots in the vein of [Morris et al 2018](http://adsabs.harvard.edu/abs/2018AJ....156...91M) and can't find the reported planet. There's more data coming in Sector 13, but we'll have to wait a while for that.

##### TIC 441462736 [b](https://archive.stsci.edu/hlsps/tess-data-alerts/hlsp_tess-data-alerts_tess_phot_00441462736-s02-01_tess_v1_dvs.pdf)

Two transits of an evolved star. Here I measure the planet radius despite the obvious spot occultation following [Morris et al 2018](http://adsabs.harvard.edu/abs/2018AJ....156...91M) and find that the exoplanet radius is probably underestimated by the DVS report due to a spot occultation in the first transit (of two total). Unfortunately, this target will not fall on silicon again according to [tess-point](https://github.com/christopherburke/tess-point).

##### TIC 149010208 [b](https://archive.stsci.edu/hlsps/tess-data-alerts/hlsp_tess-data-alerts_tess_phot_00149010208-s02-01_tess_v1_dvs.pdf)

Lots of transits of an apparently super-inflated hot Jupiter, plus two peculiar brightening events. Could this be a heartbeat star, or self-lensing non-transiting binary a la [Kruse & Agol (2014)](https://ui.adsabs.harvard.edu/#abs/2014Sci...344..275K/abstract)?

##### TIC 307210830 b/c/d [(TOI 175)](https://exo.mast.stsci.edu/exomast_planet.html?planet=TIC307210830TCE1)

M dwarf at 10 parsecs with three transiting planets.