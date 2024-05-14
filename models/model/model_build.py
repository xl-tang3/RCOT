from models.model.energy_net import EnergyNet
from models.model.upsample import UpsampleNet


def generate_model(backbone, generator, config):
    if config.name == 'Upsample':
        return UpsampleNet(backbone, generator, **config.parameters) \
            if config.parameters is not None else UpsampleNet(backbone, generator)

    if config.name == 'Energy':
        return EnergyNet(backbone, generator, **config.parameters) \
            if config.parameters is not None else EnergyNet(backbone, generator)

    else:
        print('Generator', config.name, 'Not implemented.')
        return None

