import os
import sys
import torch
from torch import nn
import torch.nn.functional as F

import librosa
#import onnx
import onnx_coreml
#import onnxruntime
import coremltools

sys.path.insert(1, os.path.join(sys.path[0], 'audioset_tagging_cnn/pytorch'))

from audioset_tagging_cnn.pytorch.pytorch_utils import interpolate, pad_framewise_output

from audioset_tagging_cnn.pytorch.models import MobileNetV1

        
class MobileNetV1Export(MobileNetV1):
    def __init__(self, *args, **kwargs):
        
        super(MobileNetV1Export, self).__init__(*args, **kwargs)
        self.interpolate_ratio = 32

        self.input_name = 'input.1'
        self.output_names = ['clip_output', 'frame_output', 'melspec' ]

        
    def forward(self, x, mixup_lambda=None):
        x = self.spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        melspec = x

        # frames_num = x.shape[2]

        # print( 'x shape: ', x.shape ) # x shape:  torch.Size([1, 1, 701, 64])
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        
        x = self.features(x)
        # print( 'features: ', x.shape ) #  (1, 1024, 3, 2)

        x = torch.mean(x, dim=3)
        # print( 'x mean: ', x.shape ) # x mean:  torch.Size([1, 1024, 3])

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        # print( x.shape, x1.shape, x2.shape )

        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        segmentwise_output = torch.sigmoid(self.fc_audioset(x))
        (clipwise_output, _) = torch.max(segmentwise_output, dim=1)

        # Get framewise output
        framewise_output = interpolate(segmentwise_output, self.interpolate_ratio)
        # DISABLED framewise_output = pad_framewise_output(framewise_output, frames_num)

        return clipwise_output, framewise_output, melspec

    def gen_torch_output( self, sample_input ):
        # Forward
        with torch.no_grad():
            raw_outputs = self( torch.from_numpy( sample_input ) )
            torch_outputs = [ item.cpu().detach().numpy() for item in raw_outputs ]
        
        for output in torch_outputs:
            print( 'out: ', output.shape )
        # out:  torch.Size([1, 96, 527])
        # out:  torch.Size([1, 1, 101, 64])

        return torch_outputs

    def convert_to_onnx( self, filename_onnx, sample_input ):

        input_names = [ self.input_name ]
        output_names = self.output_names
        
        torch.onnx.export(
            self,
            torch.from_numpy( sample_input ),
            filename_onnx,
            #input_names=input_names,
            #output_names=output_names,
            verbose=False,
            # operator_export_type=OperatorExportTypes.ONNX
        )

    def gen_onnx_outputs( self, filename_onnx, sample_input ):
        import onnxruntime
        
        session = onnxruntime.InferenceSession( filename_onnx, None)
        
        input_name = session.get_inputs()[0].name
        # output_names = [ item.name for item in session.get_outputs() ]

        raw_results = session.run([], {input_name: sample_input})

        return raw_results[0]
        
    def convert_to_coreml( self, fn_mlmodel, sample_input ):

        # first convert to ONNX
        filename_onnx = '/tmp/PANN_model.onnx'
        self.convert_to_onnx( filename_onnx, sample_input )

        # onnx_outputs = self.gen_onnx_outputs( filename_onnx, sample_input )
        
        # set up for Core ML export
        convert_params = dict(
            predicted_feature_name = [],
            minimum_ios_deployment_target='13',
            # custom_conversion_functions={'Pad':_convert_pad}, # no longer needed
        )

        mlmodel = onnx_coreml.convert(
            model=filename_onnx,
            **convert_params, 
        )

        # print(mlmodel._spec.description)

        # assert mlmodel != None, 'CoreML Conversion failed'

        mlmodel.save( fn_mlmodel )

        torch_output = self.gen_torch_output( sample_input )

        return torch_output

        """
        model_inputs = {
            self.input_name : sample_input
        }
        # do forward pass
        mlmodel_outputs = mlmodel.predict(model_inputs, useCPUOnly=True)

        # fetch the spectrogram from output dictionary
        mlmodel_output =  mlmodel_outputs[ self.output_names[0] ]
        # print( 'mlmodel_output: shape %s \nsample %s ' % ( mlmodel_output.shape, mlmodel_output[:,:,:3, :3] ) )
        print( 'mlmodel_output: shape ', ( mlmodel_output.shape ) )
        
        # mlmodel = coremltools.models.MLModel( fn_mlmodel )
        # _ = coremltools.models.MLModel( mlmodel._spec )
        """


def _convert_pad(builder, node, graph, err):
    from onnx_coreml._operators import _convert_pad as _convert_pad_orig

    pads = node.attrs['pads']
    print( 'node.name: ', node.name, pads )
    
    if node.name != 'Pad_136':
        _convert_pad_orig( builder, node, graph, err )

    else:

        params_dict = {}
        params_dict['left'] = pads[2] # padding left
        params_dict['right'] = pads[5] # padding right
        params_dict['top'] = 0
        params_dict['bottom'] = 0
        params_dict['value'] = 0.0
        params_dict['padding_type'] = 'constant'

        builder.add_padding(
            name=node.name,
            input_name=node.inputs[0],
            output_name=node.outputs[0],
            **params_dict,
        )
        


def save_class_label_json( fn_json ):
    import csv, json
    
    with open('python/audioset_tagging_cnn/metadata/class_labels_indices.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        lines = list(reader)

        labels = []
        for i1 in range(1, len(lines)):
            id = lines[i1][1]
            label = lines[i1][2]
            labels.append(label)

    with open( fn_json, 'w' ) as ofp:
        json.dump( labels, ofp )
        
def export_model( fn_mlmodel, fn_json, fn_label_json, checkpoint_path, audio_path ):

    model_args = {
        'sample_rate': 32000,
        'window_size': 1024,
        'hop_size': 320,
        'mel_bins': 64,
        'fmin': 50,
        'fmax': 14000,
        'classes_num': 527
    }
    model = MobileNetV1Export(**model_args)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Load audio
    sample_rate = model_args['sample_rate']
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
    
    sample_input = waveform[None, :sample_rate]    # (1, audio_length)
    print( 'waveform: ', sample_input.shape ) # waveform:  torch.Size([1, 224000])

    model_outputs = model.convert_to_coreml( fn_mlmodel, sample_input )

    save_model_output_as_json( fn_json, model_outputs )

    save_class_label_json( fn_label_json )

def save_model_output_as_json( fn_output, model_outputs ):
    import json
    output_data = [
        model_outputs[0][0,:].tolist(), # clipwise
        model_outputs[1][0,:].tolist(), # framewise
        model_outputs[2][0,0,:].tolist(), # melspec
    ]
    with open( fn_output, 'w' ) as fp:
        json.dump( output_data, fp )

    
if __name__ == '__main__':
    # checkpoint_path = 'MobileNetV1_mAP=0.389.pth'
    import sys
    checkpoint_path = sys.argv[1]
    audio_path = sys.argv[2]

    fn_mlmodel = '/tmp/PANN.mlmodel'
    fn_json = '/tmp/PANN_out.ring_hello.json'
    fn_label_json = '/tmp/PANN_labels.json'

    export_model( fn_mlmodel, fn_json, fn_label_json, checkpoint_path, audio_path )

# python3 python/export.py 'python/MobileNetV1_mAP=0.389.pth' '/tmp/ring_hello.wav'
# xcrun coremlc compile /tmp/PANN.mlmodel  /tmp/mlc_output


'''
import soundfile as sf

fn_wav = 'R9_ZSCveAHg_7s.wav'

waveform, samplerate = sf.read( fn_wav )
# samplerate is 32000
num_samples = 12800
sample_input = waveform[ samplerate*2:samplerate*2+num_samples ] # sec 2 to 3

sf.write( '/tmp/ring_hello.wav', sample_input, samplerate )
'''
