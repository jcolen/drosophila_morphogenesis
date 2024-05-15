
from morphogenesis.atlas_processing.ensemble_timeline import *
from morphogenesis.atlas_processing.atlas_processing import *
import pandas as pd
import os

if __name__=='__main__':
    basedir = '/Users/jcolen/Documents/drosophila_morphogenesis/flydrive/'
    savedirs = [
        'WT/Hairy/',
        'WT/Fushi_Tarazu',
        'WT/Paired',
        'WT/Sloppy_Paired',
    ]
    
    for savedir in savedirs:
        fulldir = os.path.join(basedir, savedir)
        print(savedir)
        '''
        Static datasets
        '''
        df = convert_matstruct_to_csv(fulldir, prefix='static')
        downsample_raw_tiff(fulldir)
        #We use a 20-minute morphodynamic offset for Runt
        offsets = pd.DataFrame(columns=['embryoID', 'offset']).set_index('embryoID')
        for eId in df.embryoID.unique():
            offsets.loc[eId, 'offset'] = 20
        offsets.to_csv(os.path.join(fulldir, 'morphodynamic_offsets.csv'))
        build_ensemble_timeline(fulldir, init_unc=1,
            t_min=-10, t_max=40)