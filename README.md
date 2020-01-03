# DeepIST

*DeepIST* aims to predict the travel time of a given path (i.e., a sequence of road segments) in a road network. 
Please refer the paper [here](https://arxiv.org/pdf/1909.05637.pdf).


## Prepare

### Map matching on trajectories 

    parse_[city].py <raw_fname> <output_traj_fname>
    # availeble cities: porto (update in progress)

### Parse map matching results as paths and filter incorrect matched results

    python tools/path/traj_to_path.py <traj_file> <matched_folder> <output_path_file>
    python tools/path/filter_paths.py <traj_file> <path_file> <output_path_file>

### Plot paths as images

#### 1. Prepare hourly moving speed on road segments in average based on path data

    python tools/plot/get_road_avg_speed.py <path_file> <output_speed_file>
    
#### 2. Plot paths as images

    mkdir <output_image_folder>
    python tools/plot.py <path_file> <speed_file> <osm.pbf_file> <output_image_folder> <output_training_file>

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
