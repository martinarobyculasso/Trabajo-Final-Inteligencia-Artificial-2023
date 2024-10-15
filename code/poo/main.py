import os
from CLI import CLI
import warnings


# Ignorar warnings
warnings.filterwarnings("ignore", category=UserWarning, module='librosa')
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning:librosa'

# Limpiar la consola
os.system('clear')


if __name__ == '__main__':
    CLI().cmdloop()
