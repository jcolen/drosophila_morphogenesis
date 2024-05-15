from morphogenesis.dataset import *
from morphogenesis.decomposition.decomposition_utils import build_decomposition_model


from torchvision.transforms import Compose


if __name__ == '__main__':
    transform = Reshape2DField()

    print('eCadherin dataset')
    cad_dataset = AtlasDataset('WT', 'ECad-GFP', 'raw2D', transform=Compose([transform, Smooth2D(sigma=7)]), tmin=-15, tmax=45)
    cad_vel_dataset = AtlasDataset('WT', 'ECad-GFP', 'velocity2D', transform=transform, tmin=-15, tmax=45)

    build_decomposition_model(cad_dataset, crop=10)
    build_decomposition_model(cad_vel_dataset, crop=10)

    print('Myosin dataset')
    sqh_dataset = AtlasDataset('Halo_Hetero_Twist[ey53]_Hetero', 'Sqh-GFP', 'tensor2D', transform=transform, drop_time=True, tmin=-15, tmax=45)
    sqh_vel_dataset = AtlasDataset('Halo_Hetero_Twist[ey53]_Hetero', 'Sqh-GFP', 'velocity2D', transform=transform, drop_time=True, tmin=-15, tmax=45)

    build_decomposition_model(sqh_dataset, crop=10)
    build_decomposition_model(sqh_vel_dataset, crop=10)

    print('Actin dataset')
    act_dataset = AtlasDataset('WT', 'Moesin-GFP', 'raw2D', transform=Compose([transform, LeftRightSymmetrize(), Smooth2D(sigma=7)]), tmin=-15, tmax=45)
    act_vel_dataset = AtlasDataset('WT', 'Moesin-GFP', 'velocity2D', transform=transform, tmin=-15, tmax=45)

    build_decomposition_model(act_dataset, crop=10)
    build_decomposition_model(act_vel_dataset, crop=10)

    print('Other datasets')
    runt_dataset = AtlasDataset('WT', 'Runt', 'raw2D', transform=Compose([transform, Smooth2D(sigma=3)]), tmin=-15, tmax=45)
    eve_dataset = AtlasDataset('WT', 'Even_Skipped-YFP', 'raw2D', transform=Compose([transform, Smooth2D(sigma=3)]), drop_time=True, tmin=-15, tmax=45)

    build_decomposition_model(runt_dataset, crop=10, lrsym=False)
    build_decomposition_model(eve_dataset, crop=10, lrsym=False)