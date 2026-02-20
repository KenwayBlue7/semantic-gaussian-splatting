import os; 
[os.rename(os.path.join('D:/Gaus/Buddha/input', f), os.path.join('D:/Gaus/Buddha/input', f.replace('frame_', ''))) for f in os.listdir('D:/Gaus/Buddha/input') if f.startswith('frame_')]