//
//  DrawSpecView.swift
//  CoreML_Audio_Analysis
//
//  Created by Gerald on 5/30/20.
//  Copyright © 2020 Gerald. All rights reserved.
//
// adapted from https://github.com/tucan9389/DepthPrediction-CoreML/blob/master/DepthPrediction-CoreML/DrawingHeatmapView.swift

import Foundation

import UIKit

class DrawSpecView: UIView {
    
    var spectrogram: Array<Array<Float32>>? = nil {
        didSet {
            DispatchQueue.main.async {
                self.setNeedsDisplay()
            }
        }
    }

    override func draw(_ rect: CGRect) {
        
        if let ctx = UIGraphicsGetCurrentContext() {
            
            ctx.clear(rect);
            
            guard let spectrogram = self.spectrogram else { return }
            
            let size = self.bounds.size
            let spectrogram_w = spectrogram.count
            let spectrogram_h = spectrogram.first?.count ?? 0
            let w = size.width / CGFloat(spectrogram_w)
            let h = size.height / CGFloat(spectrogram_h)
            
            for j in 0..<spectrogram_h {
                for i in 0..<spectrogram_w {
                    let value = spectrogram[i][j]
                    var alpha: CGFloat = CGFloat(value)
                    if alpha > 1 {
                        alpha = 1
                    } else if alpha < 0 {
                        alpha = 0
                    }
                    
                    let rect: CGRect = CGRect(x: CGFloat(i) * w, y: CGFloat(j) * h, width: w, height: h)
                    
                    // color
                    let hue: CGFloat = (1.0-alpha) * (240.0 / 360.0)
                    let color: UIColor = UIColor(hue: hue, saturation: 1, brightness: 1, alpha: 0.94)
                    
                    // gray
                    // let color: UIColor = UIColor(white: 1-alpha, alpha: 1)
                    
                    let bpath: UIBezierPath = UIBezierPath(rect: rect)
                    
                    color.set()
                    bpath.fill()
                }
            }
        }
    } // end of draw(rect:)

}
