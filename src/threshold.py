##############################################################################
# LABEL CLUSTERS CHAINS FOR MULTILABEL CLASSIFICATION                        #
# Copyright (C) 2025                                                         #
#                                                                            #
# This code is free software: you can redistribute it and/or modify it under #
# the terms of the GNU General Public License as published by the Free       #
# Software Foundation, either version 3 of the License, or (at your option)  #
# any later version. This code is distributed in the hope that it will be    #
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of     #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General   #
# Public License for more details.                                           #
#                                                                            #
# Prof. Elaine Cecilia Gatto - UFLA - Lavras, Minas Gerais, Brazil           #
# Prof. Ricardo Cerri - USP - São Carlos, São Paulo, Brazil                  #
# Prof. Mauri Ferrandin - UFSC - Blumenau, Santa Catarina, Brazil            #
# Prof. Celine Vens - Ku Leuven - Kortrijik, West Flanders, Belgium          #
# PhD Felipe Nakano Kenji - Ku Leuven - Kortrijik, West Flanders, Belgium    #
#                                                                            #
# BIOMAL - http://www.biomal.ufscar.br                                       #
#                                                                            #
##############################################################################



########################################################################
#                                                                      #
########################################################################
import sys
import platform
import os

system = platform.system()
if system == 'Windows':
    user_profile = os.environ['USERPROFILE']
    FolderRoot = os.path.join(user_profile, 'Documents', 'MultiLabelEvaluationMetrics', 'src')
elif system in ['Linux', 'Darwin']:  # 'Darwin' is the system name for macOS
    FolderRoot = os.path.expanduser('~/MultiLabelEvaluationMetrics/src')
else:
    raise Exception('Unsupported operating system')

os.chdir(FolderRoot)
current_directory = os.getcwd()
sys.path.append('..')


import pandas as pd
import numpy as np

