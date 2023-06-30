'''
config json utils
'''
import json 
from collections import OrderedDict

class RotateConfig:
    '''
    all configs for rotate detect method 
    TODO: modify
    '''
    def __init__(self, cfg_name=None):
        self.default_init()
        if cfg_name is not None:
            # load config
            self.load_config(cfg_name)
    
    def modify(self, args):
        '''
        modify the config by given args
        ''' 
        if args.dataset is not None:
            self._config['dataset'] = args.dataset 
        if args.backbone is not None:
            self._config['backbone'] = args.backbone
        if args.model_name is not None:
            self._config['model_name'] = args.model_name
        try: 
            if args.epoch is not None:
                self._config['epoch'] = args.epoch
            if args.batch_size is not None:
                self._config['batch_size'] = args.batch_size
            if args.gamma is not None:
                self._config['gamma'] = args.gamma
        except AttributeError:
            pass 
  
        

    @property
    def config(self):
        return self._config 

    def get(self, proper):
        return self._config.get(proper, None)

    def load_config(self, cfg_name):
        with open(cfg_name, 'rt') as fp:
            d = json.load(fp, object_hook=OrderedDict)
        for name in d:
            self._config[name] = d[name]


    def write_config(self, cfg_name):
        '''
        write all config to given json file
        ''' 
        with open(cfg_name, 'wt') as fp:
            json.dump(self._config, fp, indent=4, sort_keys=False)  

    def default_init(self): 
        self._config = {}
        # str config
        self._config['backbone'] = 'resnet50'
        self._config['dataset'] = 'Ucas'
        self._config['model_name'] = self._config['backbone'] + self._config['dataset']
        # contour config 
        self._config['contour1_scale'] = 10.0
        self._config['contour2_scale'] = 10.0
        self._config['contour3_scale'] = 10.0
        # rectangle config 
        self._config['rect1_scale'] = 100.0
        self._config['rect2_scale'] = 100.0
        self._config['rect3_scale'] = 100.0
        self._config['rect1_stride'] = 4
        self._config['rect2_stride'] = 2
        self._config['rect3_stride'] = 1 
        self._config['gamma'] = 1.0
        # config for traing process
        self._config['epoch'] = 100 
        self._config['batch_size'] = 8
        self._config['val_epoch'] = 3 
        # config for test part
        self._config['conf_thres'] = 0.5
        self._config['nms_thres'] = 0.3 # skew-iou based non-max-suppress


if __name__ == "__main__":
    cfg  = RotateConfig('test.json')
    print(cfg.config) 