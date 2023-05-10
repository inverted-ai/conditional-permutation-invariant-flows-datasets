import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import os
from infractions import get_mesh_for_map, calc_offroad
from util import recenter_offsets, rotated_tracks, location_names


class DatasetWriter():
    def __init__(self, args):
        self.args = args
        self.recenter_offsets = recenter_offsets
        self.df_cache = {}

        self.interaction_path = args.interaction_path

        self.offroad_threshold = 2.0
        self.keep_agents = 7

        self.meshes = {}

        self.name = 'fixed' if args.prune_outer else 'all'

        self.index_df = pd.read_csv(f'{self.name}_{args.split}.csv')
        self.index_df["location"] = pd.Categorical.from_codes(
            self.index_df["location_code"], categories=location_names
        )

        self.output_path = os.path.join(args.output_path, self.name, args.split)
        os.makedirs(self.output_path, exist_ok=True)

    def get_datapoint(self, location, recording, frame, center_xy=None):
        if (location, recording) in self.df_cache:
            df = self.df_cache[(location, recording)]
        else:
            df = pd.read_csv(f'{self.interaction_path}/recorded_trackfiles/{location}/vehicle_tracks_{recording:03d}.csv')
            recenter_offset = self.recenter_offsets[location]
            df['x'] = df['x'] - recenter_offset[0].item()
            df['y'] = df['y'] - recenter_offset[1].item()
            if (location, recording) in rotated_tracks:
                for track_id in rotated_tracks[(location, recording)]:
                    df.loc[df['track_id'] == track_id, 'psi_rad'] += np.pi
                    df.loc[df['track_id'] == track_id, 'psi_rad'] = df.loc[df['track_id'] == track_id, 'psi_rad'].mod(np.pi * 2)
            self.df_cache[(location, recording)] = df

        df = df.loc[df['frame_id'] == frame]

        x = torch.tensor(df[['x','y','length', 'width','psi_rad']].values).float()
        x = self.filter_offroad(location, x)
        if self.args.prune_outer:
            assert center_xy is not None
            x = self.prune_outer(x, center_xy, self.keep_agents)

        return x

    def filter_offroad(self, location, x):
        if not location in self.meshes:
            self.meshes[location] = get_mesh_for_map(location, self.interaction_path)
        mesh = self.meshes[location]
        x_off = calc_offroad(x[None, :, 0:5], mesh, self.offroad_threshold).bool()[0]

        return x[~x_off]

    def prune_outer(self, x, center_xy, keep_agents):
        x_cam = x.clone().detach()
        x_cam[:,0:2] = x_cam[:,0:2]- center_xy
        r = x_cam[:,0:2].square().sum(dim=-1).sqrt()
        ridx = torch.argsort(r)[0:keep_agents]
        return x[ridx]

    def write_dataset(self):
        for row_idx in tqdm(range(len(self.index_df))):
            _, x = self.__getitem__(row_idx)
            torch.save(x, os.path.join(self.output_path, f'{row_idx}.pt'))

    def __getitem__(self, idx):
        if self.args.prune_outer:
            loc, recording, frame, center_x, center_y = self.index_df.loc[
                idx, ["location", "recording", "initial_frame", "center_x", "center_y"]
            ]
            center_xy = torch.tensor([[center_x, center_y]])
        else:
            loc, recording, frame = self.index_df.loc[
                idx, ["location", "recording", "initial_frame"]
            ]
            center_xy = None

        return loc, self.get_datapoint(loc, recording, frame, center_xy=center_xy)


def add_arguments(parser):
    parser.add_argument('--output_path', type=str, default='output')
    parser.add_argument('--interaction_path', type=str, required=True)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--prune_outer', type=int, default=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    writer = DatasetWriter(args)
    writer.write_dataset()
