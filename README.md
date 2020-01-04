# DeepIST

*DeepIST* aims to predict the travel time of a given path (i.e., a sequence of road segments) in a road network. 
Please refer the paper [here](https://arxiv.org/pdf/1909.05637.pdf).


## Prepare Data

If you don't want to parse data from scratch by yourself, you can skip this step and use our released data.

### 1. Map matching on trajectories 

    parse_[city].py <raw_fname> <output_traj_fname>
    # for other cities: porto (well be released soon)

In this work, we apply [barefoot](https://github.com/bmwcarit/barefoot) for map matching based on [open street map(OSM)](https://www.openstreetmap.org/) data.

Some downloadable OSM data: <br/>
[portugal](https://download.geofabrik.de/europe/portugal-latest.osm.pbf) <br/>
[major cities](https://download.bbbike.org/osm/bbbike/) <br/>

Scripts we implemented for barefoot will be released soon.

### 2. Parse map matching results as paths and filter incorrect matched results

    python tools/path/traj_to_path.py <traj_file> <matched_folder> <output_path_file>
    python tools/path/filter_paths.py <traj_file> <path_file> <output_path_file>

### 3. Plot paths as images

#### 3.1. Prepare hourly moving speed on road segments in average based on path data

    python tools/plot/get_road_avg_speed.py <path_file> <output_speed_file>
    
#### 3.2. Plot paths as images

    mkdir <output_image_folder>
    python tools/plot.py <path_file> <speed_file> <osm.pbf_file> <output_image_folder> <output_training_file>
    
## Released data

### Small test data

* images [here]()
* training file [here]()

### Porto

* trajectory data [here]()
* path data [here]()
* speed data [here]()
* osm data [here](https://download.geofabrik.de/europe/portugal-latest.osm.pbf)
* images [here]()
* training file [here]()

### Chengdu

* Be released soon

## How to Use?

First, to configurations of experiments in config.py<br/>
Then, to run *DeepIST* experiments, execute the following command:<br/>

    python main.py <training_file>

## Citing

If you find *DeepIST* useful for your research, please cite the following paper:

    @inproceedings{fu2019deepist,
        title={DeepIST: Deep Image-based Spatio-Temporal Network for Travel Time Estimation},
        author={Fu, Tao-yang and Lee, Wang-Chien},
        booktitle={Proceedings of the 28th ACM International Conference on Information and Knowledge Management},
        pages={69--78},
        year={2019},
        organization={ACM}
    }

### Miscellaneous

Please send any questions you might have about the code and/or the algorithm to <txf225@cse.psu.edu> or <csiegoat@gmail.com>.
