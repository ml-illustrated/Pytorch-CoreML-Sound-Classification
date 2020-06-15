//
//  ConvertSpectrogram.swift
//  CoreML_Audio_Analysis
//
//  Created by Gerald on 5/30/20.
//  Copyright © 2020 Gerald. All rights reserved.
//
// from https://github.com/tucan9389/DepthPrediction-CoreML/blob/master/DepthPrediction-CoreML/heatmapProcessor.swift
import CoreML

class SpectrogramConverter {
    func convertTo2DArray(from spectrogram: MLMultiArray) -> Array<Array<Float32>> {
        guard spectrogram.shape.count == 4 else {
            print("spectrogram's shape is invalid. \(spectrogram.shape)")
            return []
        }
        let spectrogram_w = spectrogram.shape[2].intValue
        let spectrogram_h = spectrogram.shape[3].intValue
        
        var converted_spectrogram: Array<Array<Float32>> = Array(repeating: Array(repeating: 0.0, count: spectrogram_h), count: spectrogram_w)
        
        var minimumValue: Float32 = Float32.greatestFiniteMagnitude
        var maximumValue: Float32 = -Float32.greatestFiniteMagnitude
        
        for i in 0..<spectrogram_w {
            for j in 0..<spectrogram_h {
                let index = [ 0, 0, i as NSNumber, j as NSNumber ] // i*(spectrogram_h) + j
                let val = spectrogram[index].floatValue
                // guard val > 0 else { continue }
                converted_spectrogram[i][spectrogram_h-j-1] = val // origin at bottom
                
                if minimumValue > val {
                    minimumValue = val
                }
                if maximumValue < val {
                    maximumValue = val
                }
            }
        }
        
        maximumValue = max( -15.0, maximumValue ) // for improved contrast on device
        var minmaxGap = maximumValue - minimumValue
        
        // print( "minmax \(minmaxGap) \(maximumValue) \(minimumValue)")
        
        if ( minmaxGap == 0 ) {
            minmaxGap = 1.0
        }
        for i in 0..<spectrogram_h {
            for j in 0..<spectrogram_w {
                converted_spectrogram[j][i] = (converted_spectrogram[j][i] - minimumValue) / minmaxGap
            }
        }
        
        return converted_spectrogram
    }
}
