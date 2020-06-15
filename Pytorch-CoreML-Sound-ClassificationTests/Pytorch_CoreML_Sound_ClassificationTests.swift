//
//  Pytorch_CoreML_Sound_ClassificationTests.swift
//  Pytorch-CoreML-Sound-ClassificationTests
//
//  Created by Gerald on 6/14/20.
//  Copyright © 2020 Gerald. All rights reserved.
//

import XCTest
import AVFoundation
import CoreML

@testable import Pytorch_CoreML_Sound_Classification

class Pytorch_CoreML_Sound_ClassificationTests: XCTestCase {

    func test_labels() throws {
        
        let bundle = Bundle(for: Pytorch_CoreML_Sound_ClassificationTests.self)
        let path = bundle.path(forResource: "PANN_label", ofType: "json")

        var labels: NSArray?
        if let JSONData = try? Data(contentsOf: URL(fileURLWithPath: path!)) {
            labels = try! JSONSerialization.jsonObject(with: JSONData, options: .mutableContainers) as? NSArray
        }
        
        for i in 0..<labels!.count {
            print( "\(i) \(labels![i])")
        }

    }
    
    func test_model_inference() throws {
        
        let model = PANN()
        typealias NetworkInput = PANNInput
        typealias NetworkOutput = PANNOutput

        // read in the expected model output from JSON
        let bundle = Bundle(for: Pytorch_CoreML_Sound_ClassificationTests.self)
        let path = bundle.path(forResource: "PANN_out.ring_hello", ofType: "json")

        var expected_outputs: NSArray?
        if let JSONData = try? Data(contentsOf: URL(fileURLWithPath: path!)) {
            expected_outputs = try! JSONSerialization.jsonObject(with: JSONData, options: .mutableContainers) as? NSArray
        }
        let expected_clipwise: [NSNumber] = expected_outputs![0] as! [NSNumber]
        print( "expected clipwise: \(expected_clipwise.count)" )

        let expected_spectrogram: [[NSNumber]] = expected_outputs![2] as! [[NSNumber]]
        print( "expected spec: \(expected_spectrogram.count) \(expected_spectrogram[0].count)")

        // read the input shapes of our model
        let inputName = "input.1"
        let inputConstraint: MLFeatureDescription = model.model.modelDescription
            .inputDescriptionsByName[inputName]!

        let input_batch_size: Int = Int(truncating: (inputConstraint.multiArrayConstraint?.shape[0])! )
        let input_samples: Int = Int(truncating: (inputConstraint.multiArrayConstraint?.shape[1])! )
        // print( "inputCon: \(input_batch_size) \(input_samples)")

        // read the same WAV file used in PyTorch
        let testBundle = Bundle(for: type(of: self))
        guard let filePath = testBundle.path(forResource: "ring_hello", ofType: "wav") else {
            fatalError( "error opening ring_hello.wav" )
        }
        
        // Read wav file
        var wav_file:AVAudioFile!
        do {
           let fileUrl = URL(fileURLWithPath: filePath )
           wav_file = try AVAudioFile( forReading:fileUrl )
        } catch {
           fatalError("Could not open wav file.")
        }

        print("wav file length: \(wav_file.length)")

        let buffer = AVAudioPCMBuffer(pcmFormat: wav_file.processingFormat,
                                      frameCapacity: UInt32(wav_file.length))
        do {
           try wav_file.read(into:buffer!)
        } catch{
           fatalError("Error reading buffer.")
        }
        
        guard let bufferData = buffer?.floatChannelData![0] else {
           fatalError("Can not get a float handle to buffer")
        }
            

        // allocate a ML Array & copy samples over
        let array_shape = [input_batch_size as NSNumber, input_samples as NSNumber]
        let audioData = try! MLMultiArray(shape: array_shape, dataType: MLMultiArrayDataType.float32 )
        let ptr = UnsafeMutablePointer<Float32>(OpaquePointer(audioData.dataPointer))
        for i in 0..<input_samples {
            ptr[i] = Float32(bufferData[i])
        }

        // create the input dictionary as { 'input.1' : [<wave floats>] }
        let inputs: [String: Any] = [
            inputName: audioData,
        ]
        // container for ML Model inputs
        let provider = try! MLDictionaryFeatureProvider(dictionary: inputs)
        
        // Send the wavefor samples into the model to generate the Spectrogram
        let raw_outputs = try! model.model.prediction(from: provider)
        
        // convert raw dictionary into our model's output object
        let outputs = NetworkOutput( features: raw_outputs )

        
        let output_clipwise: MLMultiArray = outputs.clip_output
        print( "clip outputs: \(output_clipwise.shape)") // [1, 527]

        // sanity check the shapes of our output
        XCTAssertTrue( Int( truncating: output_clipwise.shape[1] ) == expected_clipwise.count,
                       "incorrect shape[1]! \(output_clipwise.shape[1]) \(expected_clipwise.count)" )

        
        // compare every element of our spectrogram with those from the JSON file
        for i in 0..<expected_clipwise.count {
            let test_idx: [NSNumber] = [ 0, NSNumber(value: i) ]
            let val = output_clipwise[ test_idx ].floatValue
            // print( "clipwise vals \(i) \(val), \(expected_clipwise[i].floatValue)" )
            
            XCTAssertLessThan( abs( val - expected_clipwise[i].floatValue ), 0.1,
                               "clipwise vals different at \(i)! \(val), \(expected_clipwise[i].floatValue)" )
            
        }
        
        
        let output_spectrogram: MLMultiArray = outputs.melspec
        print( "outputs: \(output_spectrogram.shape)") // [1, 1, 28, 64]

        // sanity check the shapes of our output
        XCTAssertTrue( Int( truncating: output_spectrogram.shape[2] ) == expected_spectrogram.count,
                       "incorrect shape[2]! \(output_spectrogram.shape[2]) \(expected_spectrogram.count)" )
        XCTAssertTrue( Int( truncating: output_spectrogram.shape[3] ) == expected_spectrogram[0].count,
                       "incorrect shape[3]! \(output_spectrogram.shape[3]) \(expected_spectrogram[0].count)" )

        // compare every element of our spectrogram with those from the JSON file
        for i in 0..<expected_spectrogram.count {
            let spec_row = expected_spectrogram[i] as [NSNumber]

            for j in 0..<spec_row.count {
                let test_idx: [NSNumber] = [ 0, 0, NSNumber(value: i), NSNumber(value: j) ]
                let val = output_spectrogram[ test_idx ].floatValue
                XCTAssertLessThan( abs( val - spec_row[j].floatValue ), 15,
                                   "spec vals different at \(i) \(j)! \(val), \(spec_row[j].floatValue)" )
            }
        }
    }

    func testPerformanceExample() throws {
        // This is an example of a performance test case.
        self.measure {
            // Put the code you want to measure the time of here.
        }
    }

}
