import copy

import h5py
import numpy as np
import torch
from scOT.problems.base import BaseTimeDataset
from scOT.problems.fluids.normalization_constants import CONSTANTS


class IncompressibleBase(BaseTimeDataset):
    def __init__(
        self,
        N_max,
        file_path,
        *args,
        tracer=False,
        just_velocities=False,
        transpose=False,
        resolution=None,
        **kwargs,
    ):
        """
        just_velocities: If True, only the velocities are used as input and output.
        transpose: If True, the input and output are transposed.
        """
        super().__init__(*args, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 20

        self.N_max = N_max
        self.N_val = 120
        self.N_test = 240
        self.resolution = 128
        self.tracer = tracer
        self.just_velocities = just_velocities
        self.transpose = transpose

        data_path = self.data_path + file_path
        data_path = self._move_to_local_scratch(data_path)
        self.reader = h5py.File(data_path, "r")

        self.constants = copy.deepcopy(CONSTANTS)
        if just_velocities:
            self.constants["mean"] = self.constants["mean"][1:3]
            self.constants["std"] = self.constants["std"][1:3]

        self.density = torch.ones(1, self.resolution, self.resolution)
        self.pressure = torch.zeros(1, self.resolution, self.resolution)

        self.input_dim = 4 if not tracer else 5
        if just_velocities:
            self.input_dim -= 2
        self.label_description = "[u,v]"
        if not self.just_velocities:
            self.label_description = "[rho],[u,v],[p]"
        if tracer:
            self.label_description += ",[tracer]"

        self.pixel_mask = torch.tensor([False, False])
        if not self.just_velocities:
            self.pixel_mask = torch.tensor([False, False, False, True])
        if tracer:
            self.pixel_mask = torch.cat(
                [self.pixel_mask, torch.tensor([False])],
                dim=0,
            )

        if resolution is None:
            self.res = None
        else:
            if resolution > 128:
                raise ValueError("Resolution must be <= 128")
            self.res = resolution

        self.post_init()

    def _downsample(self, image, target_size):
        image = image.unsqueeze(0)
        image_size = image.shape[-2]
        freqs = torch.fft.fftfreq(image_size, d=1 / image_size)
        sel = torch.logical_and(freqs >= -target_size / 2, freqs <= target_size / 2 - 1)
        image_hat = torch.fft.fft2(image, norm="forward")
        image_hat = image_hat[:, :, sel, :][:, :, :, sel]
        image = torch.fft.ifft2(image_hat, norm="forward").real
        return image.squeeze(0)

    # def __getitem__(self, idx):
    #     i, t, t1, t2 = self._idx_map(idx)
    #     time = t / self.constants["time"]

    #     inputs_v = (
    #         torch.from_numpy(self.reader["velocity"][i + self.start, t1, 0:2])
    #         .type(torch.float32)
    #         .reshape(2, self.resolution, self.resolution)
    #     )
    #     label_v = (
    #         torch.from_numpy(self.reader["velocity"][i + self.start, t2, 0:2])
    #         .type(torch.float32)
    #         .reshape(2, self.resolution, self.resolution)
    #     )
    #     if self.transpose:
    #         inputs_v = inputs_v.transpose(-2, -1)
    #         label_v = label_v.transpose(-2, -1)

    #     if not self.just_velocities:
    #         inputs = torch.cat([self.density, inputs_v, self.pressure], dim=0)
    #         label = torch.cat([self.density, label_v, self.pressure], dim=0)
    #     else:
    #         inputs = inputs_v
    #         label = label_v

    #     inputs = (inputs - self.constants["mean"]) / self.constants["std"]
    #     label = (label - self.constants["mean"]) / self.constants["std"]

    #     if self.tracer:
    #         input_tracer = (
    #             torch.from_numpy(self.reader["velocity"][i + self.start, t1, 2:3])
    #             .type(torch.float32)
    #             .reshape(1, self.resolution, self.resolution)
    #         )
    #         output_tracer = (
    #             torch.from_numpy(self.reader["velocity"][i + self.start, t2, 2:3])
    #             .type(torch.float32)
    #             .reshape(1, self.resolution, self.resolution)
    #         )
    #         if self.transpose:
    #             input_tracer = input_tracer.transpose(-2, -1)
    #             output_tracer = output_tracer.transpose(-2, -1)
    #         input_tracer = (
    #             input_tracer - self.constants["tracer_mean"]
    #         ) / self.constants["tracer_std"]
    #         output_tracer = (
    #             output_tracer - self.constants["tracer_mean"]
    #         ) / self.constants["tracer_std"]

    #         inputs = torch.cat([inputs, input_tracer], dim=0)
    #         label = torch.cat([label, output_tracer], dim=0)

    #     if self.res is not None:
    #         inputs = self._downsample(inputs, self.res)
    #         label = self._downsample(label, self.res)

    #     return {
    #         "pixel_values": inputs,
    #         "labels": label,
    #         "time": time,
    #         "pixel_mask": self.pixel_mask,
    #     }
    
    def __getitem__(self, idx):
        """
        Returns the entire video (all timesteps) for a given index.
        """
        i = idx
        video = []
        for t in range(self.max_num_time_steps):
            frame = torch.from_numpy(self.reader["velocity"][i + self.start, t, 0:2])
            frame = frame.type(torch.float32).reshape(2, self.resolution, self.resolution)
            if self.transpose:
                frame = frame.transpose(-2, -1)
            frame = (frame - self.constants["mean"]) / self.constants["std"]
            video.append(frame)
        video = torch.stack(video, dim=0)  # Shape: (T, C, H, W)

        # # Debugging: Print shapes and values
        # print("self.max_num_time_steps", self.max_num_time_steps)
        # print("video.shape", video.shape)
        # print("self.resolution", self.resolution)
        # print("self.constants['mean']", self.constants["mean"])
        # print("self.constants['std']", self.constants["std"])
        # # input("Press Enter to continue...")

        # Optionally include additional data like density, pressure, or tracer
        if not self.just_velocities:
            density = self.density.unsqueeze(0).repeat(self.max_num_time_steps, 1, 1, 1)
            pressure = self.pressure.unsqueeze(0).repeat(self.max_num_time_steps, 1, 1, 1)
            video = torch.cat([density, video, pressure], dim=1)  # Shape: (T, C, H, W)
        else:
            pressure = self.pressure.unsqueeze(0).repeat(self.max_num_time_steps, 1, 1, 1)
            video = torch.cat([video, pressure], dim=1)  # Shape: (T, C, H, W)

        if self.tracer:
            tracer_video = []
            for t in range(self.max_num_time_steps):
                tracer_frame = torch.from_numpy(self.reader["velocity"][i + self.start, t, 2:3])
                tracer_frame = tracer_frame.type(torch.float32).reshape(1, self.resolution, self.resolution)
                if self.transpose:
                    tracer_frame = tracer_frame.transpose(-2, -1)
                tracer_frame = (tracer_frame - self.constants["tracer_mean"]) / self.constants["tracer_std"]
                tracer_video.append(tracer_frame)
            tracer_video = torch.stack(tracer_video, dim=0)  # Shape: (T, 1, H, W)
            video = torch.cat([video, tracer_video], dim=1)  # Shape: (T, C, H, W)

        if self.res is not None:
            video = torch.stack([self._downsample(frame, self.res) for frame in video], dim=0)

        return video, torch.zeros(1)


class IncompressibleVideoBase(IncompressibleBase):
    def __init__(self, *args, **kwargs):
        """
        Inherits from IncompressibleBase and adds functionality to return entire videos.
        """
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        """
        Returns the entire video (all timesteps) for a given index.
        """
        i = idx
        video = []
        for t in range(self.max_num_time_steps):
            frame = torch.from_numpy(self.reader["velocity"][i + self.start, t, 0:2])
            frame = frame.type(torch.float32).reshape(2, self.resolution, self.resolution)
            if self.transpose:
                frame = frame.transpose(-2, -1)
            frame = (frame - self.constants["mean"]) / self.constants["std"]
            video.append(frame)
        video = torch.stack(video, dim=0)  # Shape: (T, C, H, W)

        # Optionally include additional data like density, pressure, or tracer
        if not self.just_velocities:
            density = self.density.unsqueeze(0).repeat(self.max_num_time_steps, 1, 1, 1)
            pressure = self.pressure.unsqueeze(0).repeat(self.max_num_time_steps, 1, 1, 1)
            video = torch.cat([density, video, pressure], dim=1)  # Shape: (T, C, H, W)

        if self.tracer:
            tracer_video = []
            for t in range(self.max_num_time_steps):
                tracer_frame = torch.from_numpy(self.reader["velocity"][i + self.start, t, 2:3])
                tracer_frame = tracer_frame.type(torch.float32).reshape(1, self.resolution, self.resolution)
                if self.transpose:
                    tracer_frame = tracer_frame.transpose(-2, -1)
                tracer_frame = (tracer_frame - self.constants["tracer_mean"]) / self.constants["tracer_std"]
                tracer_video.append(tracer_frame)
            tracer_video = torch.stack(tracer_video, dim=0)  # Shape: (T, 1, H, W)
            video = torch.cat([video, tracer_video], dim=1)  # Shape: (T, C, H, W)

        if self.res is not None:
            video = torch.stack([self._downsample(frame, self.res) for frame in video], dim=0)

        return video


class KolmogorovFlow(BaseTimeDataset):
    def __init__(self, *args, tracer=False, just_velocities=False, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 20

        assert tracer == False

        self.N_max = 20000
        self.N_val = 120
        self.N_test = 240
        self.resolution = 128
        self.just_velocities = just_velocities

        data_path = self.data_path + "/FNS-KF.nc"
        data_path = self._move_to_local_scratch(data_path)
        self.reader = h5py.File(data_path, "r")

        self.constants = copy.deepcopy(CONSTANTS)
        self.constants["mean"][1] = -2.2424793e-13
        self.constants["mean"][2] = 4.1510376e-12
        self.constants["std"][1] = 0.22017328
        self.constants["std"][2] = 0.22078253
        if just_velocities:
            self.constants["mean"] = self.constants["mean"][1:3]
            self.constants["std"] = self.constants["std"][1:3]

        self.density = torch.ones(1, self.resolution, self.resolution)
        self.pressure = torch.zeros(1, self.resolution, self.resolution)
        X, Y = torch.meshgrid(
            torch.linspace(0, 1, self.resolution),
            torch.linspace(0, 1, self.resolution),
            indexing="ij",
        )
        f = lambda x, y: 0.1 * torch.sin(2.0 * np.pi * (x + y))
        self.forcing = f(X, Y).unsqueeze(0)
        self.constants["mean_forcing"] = -1.2996679288335145e-09
        self.constants["std_forcing"] = 0.0707106739282608
        self.forcing = (self.forcing - self.constants["mean_forcing"]) / self.constants[
            "std_forcing"
        ]

        self.input_dim = 5 if not tracer else 6
        if just_velocities:
            self.input_dim -= 2
        self.label_description = "[u,v],[g]"
        if not self.just_velocities:
            self.label_description = "[rho],[u,v],[p],[g]"
        if tracer:
            self.label_description += ",[tracer]"

        self.pixel_mask = torch.tensor([False, False, False])
        if not self.just_velocities:
            self.pixel_mask = torch.tensor([False, False, False, True, False])
        if tracer:
            self.pixel_mask = torch.cat(
                [self.pixel_mask, torch.tensor([False])],
                dim=0,
            )

        self.post_init()

    def __getitem__(self, idx):
        i, t, t1, t2 = self._idx_map(idx)
        time = t / self.constants["time"]

        inputs_v = (
            torch.from_numpy(self.reader["solution"][i + self.start, t1, 0:2])
            .type(torch.float32)
            .reshape(2, self.resolution, self.resolution)
        )
        label_v = (
            torch.from_numpy(self.reader["solution"][i + self.start, t2, 0:2])
            .type(torch.float32)
            .reshape(2, self.resolution, self.resolution)
        )

        if not self.just_velocities:
            inputs = torch.cat([self.density, inputs_v, self.pressure], dim=0)
            label = torch.cat([self.density, label_v, self.pressure], dim=0)
        else:
            inputs = inputs_v
            label = label_v

        inputs = (inputs - self.constants["mean"]) / self.constants["std"]
        label = (label - self.constants["mean"]) / self.constants["std"]

        inputs = torch.cat([inputs, self.forcing], dim=0)
        label = torch.cat([label, self.forcing], dim=0)

        return {
            "pixel_values": inputs,
            "labels": label,
            "time": time,
            "pixel_mask": self.pixel_mask,
        }


class BrownianBridge(IncompressibleBase):
    def __init__(self, *args, tracer=False, just_velocities=False, **kwargs):
        if tracer:
            raise ValueError("BrownianBridge does not have a tracer")
        file_path = "/NS-BB.nc"
        super().__init__(
            20000,
            file_path,
            *args,
            tracer=False,
            just_velocities=just_velocities,
            **kwargs,
        )


class PiecewiseConstants(IncompressibleBase):
    def __init__(self, *args, tracer=False, just_velocities=False, **kwargs):
        file_path = "/NS-PwC.nc"
        super().__init__(
            20000,
            file_path,
            *args,
            tracer=tracer,
            just_velocities=just_velocities,
            **kwargs,
        )


class Gaussians(IncompressibleBase):
    def __init__(self, *args, tracer=False, just_velocities=False, **kwargs):
        if tracer:
            raise ValueError("Gaussians does not have a tracer")
        file_path = "/NS-Gauss.nc"
        super().__init__(
            20000,
            file_path,
            *args,
            tracer=False,
            just_velocities=just_velocities,
            **kwargs,
        )


class ShearLayer(IncompressibleBase):
    def __init__(self, *args, tracer=False, just_velocities=False, **kwargs):
        if tracer:
            raise ValueError("Shear layer does not have a tracer")
        super().__init__(
            40000,
            "/NS-SL.nc",
            *args,
            transpose=True,
            tracer=False,
            just_velocities=just_velocities,
            **kwargs,
        )


class VortexSheet(IncompressibleBase):
    def __init__(self, *args, tracer=False, just_velocities=False, **kwargs):
        if tracer:
            raise ValueError("VortexSheet does not have a tracer")
        file_path = "/NS-SVS.nc"
        super().__init__(
            20000,
            file_path,
            *args,
            tracer=False,
            just_velocities=just_velocities,
            **kwargs,
        )


class Sines(IncompressibleBase):
    def __init__(self, *args, tracer=False, just_velocities=False, **kwargs):
        if tracer:
            raise ValueError("Sines does not have a tracer")
        file_path = "/NS-Sines.nc"
        super().__init__(
            20000,
            file_path,
            *args,
            tracer=False,
            just_velocities=just_velocities,
            **kwargs,
        )
