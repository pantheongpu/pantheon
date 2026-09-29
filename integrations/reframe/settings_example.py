# A ReFrame configuration for trying pantheon_check.py on one machine.
#
#   reframe -C settings_example.py -c pantheon_check.py -r --exec-policy serial
#
# A site that runs ReFrame already has its own configuration. What the test
# needs from it is in the two lines marked below.

site_configuration = {
    'systems': [
        {
            'name': 'workstation',
            'descr': 'One machine with GPUs, no scheduler',
            'hostnames': ['.*'],
            'partitions': [
                {
                    'name': 'gpu',
                    'scheduler': 'local',
                    'launcher': 'local',
                    'environs': ['gpu-toolchain'],
                    'features': ['gpu'],                       # needed by the test
                    'devices': [{'type': 'gpu', 'num_devices': 1}],
                }
            ],
        }
    ],
    'environments': [
        {
            'name': 'gpu-toolchain',
            'features': ['cuda'],                              # 'hip' on AMD cards
        }
    ],
}
