# AutomaticCategoricalFeatureBinning

We don't support column "MSSubClass" of data houses dataset since it's ordinal variable.

Right now only tried on column "gill-color" of mushrooms dataset.

frequency=0.01, bin_score=0.59, bins=[['9', '11']]
frequency=0.01, bin_score=0.59, bins=[['9', '11', '10']]
frequency=0.01, bin_score=0.59, bins=[['9', '11', '10', '7']]
frequency=0.05, bin_score=0.59, bins=[['9', '11', '10', '7', '0']]
frequency=0.06, bin_score=0.59, bins=[['9', '11', '10', '7', '0', '6']]
frequency=0.09, bin_score=0.59, bins=[['9', '5'], ['11', '10', '7', '0', '6']]
frequency=0.09, bin_score=0.59, bins=[['9', '5', '2'], ['11', '10', '7', '0', '6']]
frequency=0.13, bin_score=0.59, bins=[['9', '5', '2'], ['11', '10', '7', '0', '6', '1']]
frequency=0.15, bin_score=0.59, bins=[['9', '5', '2'], ['11', '10', '7', '0', '6', '1', '4']]
frequency=0.18, bin_score=0.60, bins=[['9', '5', '2'], ['11', '10', '7', '0', '6', '1', '4', '3']]