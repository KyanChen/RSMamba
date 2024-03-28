from typing import List, Optional, Union

import mmengine

from mmpretrain.registry import DATASETS
from ...datasets import BaseDataset

CATEGORIES = {
	'AID': ["Port", "Square", "River", "Pond", "Playground", "Mountain", "Park", "Airport", "SparseResidential", "Stadium", "Industrial", "Church", "DenseResidential", "Parking", "Farmland", "Desert", "Viaduct", "Forest", "StorageTanks", "Center", "Meadow", "MediumResidential", "School", "BareLand", "Commercial", "BaseballField", "RailwayStation", "Resort", "Bridge", "Beach"],
	'UC': ["harbor", "parkinglot", "buildings", "mediumresidential", "sparseresidential", "forest", "agricultural", "overpass", "golfcourse", "freeway", "baseballdiamond", "beach", "intersection", "runway", "mobilehomepark", "chaparral", "airplane", "river", "storagetanks", "tenniscourt", "denseresidential"],
	'NWPU': ["river", "terrace", "snowberg", "stadium", "parking_lot", "thermal_power_station", "rectangular_farmland", "cloud", "intersection", "meadow", "palace", "ground_track_field", "commercial_area", "beach", "church", "baseball_diamond", "ship", "industrial_area", "bridge", "freeway", "medium_residential", "circular_farmland", "mobile_home_park", "railway", "forest", "sparse_residential", "harbor", "wetland", "mountain", "airport", "golf_course", "overpass", "dense_residential", "chaparral", "basketball_court", "airplane", "tennis_court", "runway", "storage_tank", "island", "railway_station", "lake", "sea_ice", "desert", "roundabout"]
}

@DATASETS.register_module()
class RSClsDataset(BaseDataset):
	def __init__(self,
	             data_root: str = '',
	             data_name: str = 'AID',
	             ann_file: str = '',
	             **kwargs):
		metainfo = {'classes': CATEGORIES[data_name]}
		super().__init__(
			data_root=data_root,
			ann_file=ann_file,
			metainfo=metainfo,
			**kwargs)

	def load_data_list(self) -> List[dict]:
		img_files = mmengine.list_from_file(self.ann_file)
		data_list = []
		for img_file in img_files:
			img_info = dict(
				img_path=self.data_root + '/' + img_file,
				gt_label=self.class_to_idx[img_file.split('/')[0]]
			)
			data_list.append(img_info)
		return data_list

	def prepare_data(self, idx):
		data_info = self.get_data_info(idx)
		results = self.pipeline(data_info)
		return results
