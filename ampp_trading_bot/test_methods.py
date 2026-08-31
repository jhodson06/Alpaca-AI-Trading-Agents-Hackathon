import inspect
from alpaca.data.live.option import OptionDataStream

print("Methods in OptionDataStream:")
for name, obj in inspect.getmembers(OptionDataStream):
    if name.startswith("subscribe"):
        print(name)
