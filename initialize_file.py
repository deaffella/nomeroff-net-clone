import os
import numpy as np
import sys
import matplotlib.image as mpimg
import tensorflow as tf

from NomeroffNet import filters, RectDetector, TextDetector, OptionsDetector, Detector, textPostprocessing, \
    textPostprocessingAsync

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if __name__=="__main__":


    print('npdetector with default configuration file')
    nnet = Detector()
    nnet.loadModel("latest")


    print('Initialize Rect Detector')
    rectDetector = RectDetector()

    print('Initialize Options Detector')
    optionsDetector = OptionsDetector()
    optionsDetector.load("latest")

    print('Initialize text detector')
    textDetector = TextDetector({
        "eu_ua_2004_2015": {
            "for_regions": ["eu_ua_2015", "eu_ua_2004"],
            "model_path": "latest"
        },
        "eu_ua_1995": {
            "for_regions": ["eu_ua_1995"],
            "model_path": "latest"
        },
        "eu": {
            "for_regions": ["eu"],
            "model_path": "latest"
        },
        "ru": {
            "for_regions": ["ru", "eu-ua-fake-lnr", "eu-ua-fake-dnr"],
            "model_path": "latest"
        },
        "kz": {
            "for_regions": ["kz"],
            "model_path": "latest"
        },
        "ge": {
            "for_regions": ["ge"],
            "model_path": "latest"
        }
    })

    img_path = '/var/www/1.JPG'
    print(img_path)

    img = mpimg.imread(img_path)
    NP = nnet.detect([img])

    # Generate image mask.
    cv_img_masks = filters.cv_img_mask(NP)

    # Detect points.
    arrPoints = rectDetector.detect(cv_img_masks)
    zones = rectDetector.get_cv_zonesBGR(img, arrPoints)

    # find standart
    regionIds, stateIds, countLines = optionsDetector.predict(zones)
    regionNames = optionsDetector.getRegionLabels(regionIds)

    # find text with postprocessing by standart
    textArr = textDetector.predict(zones)
    textArr = textPostprocessing(textArr, regionNames)
    print(textArr)