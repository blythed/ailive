def map_legacy_parameters(cf):
    new_cf = {'discriminator': {}, 'generator': {}}

    if 'nz' in cf['generator']:
        new_cf['generator']['in_channels'] = new_cf['generator']['nz']
    if 'Ctype' in cf['generator']:
        new_cf['generator']['convolution_type'] = \
            {1: 'regular', 0: 'transpose'}[cf['generator']['Ctype']]
    if 'nDep' in cf['generator']:
        new_cf['generator']['network_depth'] = cf['generator']['nDep']
    if 'nc' in cf['generator']:
        new_cf['generator']['image_channels'] = cf['generator']['nc']
    if 'ngf' in cf['generator']:
        new_cf['generator']['channel_number_parameter'] = cf['generator']['ngf']

    if 'ndf' in cf['discriminator']:
        new_cf['discriminator']['channel_number_parameter'] = new_cf['discriminator']['ndf']
    if 'nDep' in cf['discriminator']:
        new_cf['discriminator']['network_depth'] = cf['discriminator']['nDep']
    if 'nc' in cf['discriminator']:
        new_cf['discriminator']['image_channels'] = cf['discriminator']['nc']
    if 'Ctype' in cf['discriminator']:
        new_cf['discriminator']['convolution_type'] = \
            {1: 'regular', 0: 'transpose'}[cf['discriminator']['Ctype']]
    if 'bOut1' in cf['discriminator']:
        new_cf['discriminator']['predict_binary'] = cf['discriminator']['bOut1']

    return new_cf
