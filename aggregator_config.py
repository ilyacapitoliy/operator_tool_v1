AGGREGATOR_DEFINITIONS = {
    "SBM":                ["Size", "Brand", "Model"],
    "Model":              ["Model"],
    "Brand":              ["Brand"],
    "Size":               ["Size"],
    "SizeBrand":          ["Size", "Brand"],
    "BrandModel":         ["Brand", "Model"],
    "LoadSpeed":          ["LoadSpeed"],
    "SBMLoadSpeed":       ["Size", "Brand", "Model", "LoadSpeed"],
    "SizeLoadSpeed":      ["Size", "LoadSpeed"],
    "BrandLoadSpeed":     ["Brand", "LoadSpeed"],
    "SizeBrandLoadSpeed": ["Size", "Brand", "LoadSpeed"],
    "BrandModelLoadSpeed":["Brand", "Model", "LoadSpeed"]
}

AGGREGATOR_PRIORITY_WEIGHTS = {
    "SBMLoadSpeed":        1.00,
    "SBM":                 0.90,
    "SizeBrandLoadSpeed":  0.70,
    "SizeBrand":           0.60,
    "BrandModelLoadSpeed": 0.70,
    "BrandModel":          0.60,
    "SizeLoadSpeed":       0.60,
    "Size":                0.30,
    "LoadSpeed":           0.30,
    "BrandLoadSpeed":      0.50,
    "Brand":               0.40,
    "Model":               0.45
}