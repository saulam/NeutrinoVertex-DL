"""
Project: "Deep-learning-based decomposition of overlapping-sparse images:
          application at the vertex of neutrino interactions"
Paper: https://arxiv.org/abs/2310.19695.
Author: Dr. Saul Alonso-Monsalve
Contact: salonso@ethz.ch/saul.alonso.monsalve@cern.ch
Description: Script with auxiliary functions.
"""

import numpy as np
import torch


def set_random_seed(seed, random=None, numpy=None, torch=None):
    """
    Set random number generator seeds for NumPy and Python's random module.

    Args:
        seed (int): The seed value to ensure reproducibility in random number generation.
        random (module, optional): The Python random module. Defaults to None.
        numpy (module, optional): The NumPy module. Defaults to None.
        torch (module, optional): The PyTorch module. Defaults to None.
    """
    if random:
        random.seed(seed)
    if numpy:
        numpy.random.seed(seed)
    if torch:
        torch.manual_seed(seed)


def calc_cont_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Given points a and b, calculate point c so ab and bc have the same direction and length
    and they follow the same straight line so a->b->c

    Args:
        a (numpy.ndarray): A 3D point representing the starting point.
        b (numpy.ndarray): A 3D point representing the intermediate point.

    Returns:
        c (numpy.ndarray): A 3D point that extends the line formed by 'a' and 'b'.

    Raises:
        ValueError: If points 'a' and 'b' are the same, 'c' cannot be calculated.
    """
    # Calculate the vector AB from A to B
    ab = b - a
    # Calculate the length of vector AB
    len_ab = np.linalg.norm(ab)
    if len_ab == 0:
        raise ValueError("Points a and b are the same, cannot calculate continuation.")
    # Normalise the vector AB to get a unit vector
    unit_vec_ab = ab / len_ab
    # Calculate point C by adding the vector AB to point B
    c = b + unit_vec_ab * len_ab
    return c


def shift_image(image, shift_x, shift_y, shift_z, img_size):
    """
    Shift a 3D image by specified offsets in the x, y, and z dimensions.

    Args:
        image (numpy.ndarray): The input 3D image to be shifted.
        shift_x (int): The shift offset in the x-dimension.
        shift_y (int): The shift offset in the y-dimension.
        shift_z (int): The shift offset in the z-dimension.
        img_size (int): The size of the 3D image in each dimension.

    Returns:
        shifted_image (numpy.ndarray): The shifted 3D image.
    """
    inner_size = img_size
    outer_size = inner_size + 2

    # Shift image
    src_x_slice = slice(max(0, -shift_x), min(outer_size, outer_size - shift_x))
    src_y_slice = slice(max(0, -shift_y), min(outer_size, outer_size - shift_y))
    src_z_slice = slice(max(0, -shift_z), min(outer_size, outer_size - shift_z))
    dst_x_slice = slice(max(0, shift_x), min(outer_size, outer_size + shift_x))
    dst_y_slice = slice(max(0, shift_y), min(outer_size, outer_size + shift_y))
    dst_z_slice = slice(max(0, shift_z), min(outer_size, outer_size + shift_z))
    shifted_image = np.zeros_like(image)
    shifted_image[dst_x_slice, dst_y_slice, dst_z_slice] = image[src_x_slice, src_y_slice, src_z_slice]
    return shifted_image


def shift_particle(particle_pos, shift_x, shift_y, shift_z, cube_size):
    """
    Shift a particle position by specified offsets in the x, y, and z dimensions.

    Args:
        particle_pos (list or numpy.ndarray): The 3D position of the particle as [x, y, z].
        shift_x (int): The shift offset in the x-dimension.
        shift_y (int): The shift offset in the y-dimension.
        shift_z (int): The shift offset in the z-dimension.
        cube_size (float): The size of the cube or voxel representing the spatial coordinates.
    """
    particle_pos[0] += (shift_x * cube_size)
    particle_pos[1] += (shift_y * cube_size)
    particle_pos[2] += (shift_z * cube_size)


def fix_exit_shift(pos_exit, pos_exit_reduce, shift_x, shift_y, shift_z):
    """
    Adjust the exit point of a muon particle considering a potential random shift.

    Args:
        pos_exit (numpy.ndarray): The original exit point of the muon as a 3D position.
        pos_exit_reduce (numpy.ndarray): The reduced exit point of the muon.
        shift_x (int): The shift offset in the x-dimension.
        shift_y (int): The shift offset in the y-dimension.
        shift_z (int): The shift offset in the z-dimension.

    Returns:
        numpy.ndarray: The adjusted exit point of the muon.
        bool: A flag indicating whether a shift was made towards the exit plane.
    """
    # Calculate the plane (x, y, z) the muon is exiting
    exit_plane = np.argwhere(np.abs(pos_exit) >= 45)[0, 0]
    shift = np.array([shift_x, shift_y, shift_z])

    # Check whether there's a random shift towards the exiting plane
    pos_exit_target = None
    if np.abs(shift).sum() > 0 and abs(shift[exit_plane]) > 0:
        # Case A: shifting towards the exiting plane -> calculate new exiting point
        if ((pos_exit[exit_plane] > 0) and (shift[exit_plane] > 0)) or \
                ((pos_exit[exit_plane] < 0) and (shift[exit_plane] < 0)):
            pos_exit_target = calc_cont_point(pos_exit, pos_exit_reduce)
        # Case B: shifting opposite direction to the exiting plane ->
        #         new exiting point is the one from outer volume
        elif ((pos_exit[exit_plane] > 0) and (shift[exit_plane] < 0)) or \
                ((pos_exit[exit_plane] < 0) and (shift[exit_plane] > 0)):
            pos_exit_target = pos_exit.copy()
        shift_plane = True
    else:
        # There is no random shift or the shift is done not towards the exiting plane
        pos_exit_target = pos_exit_reduce.copy()
        shift_plane = False

    return pos_exit_target, shift_plane


def fix_empty_particles(counts_p, counts_D, counts_T, np_rand_obj):
    """
    Adjust event compositions by fixing empty counts with one random particle assignment.

    Args:
        counts_p (numpy.ndarray): An array of proton counts per event.
        counts_D (numpy.ndarray): An array of deuterium counts per event.
        counts_T (numpy.ndarray): An array of tritium counts per event.
        np_rand_obj (numpy.random.RandomState): A NumPy random state object for random number generation.
    """
    # check if events have 0 particles and fix
    particles = counts_p + counts_D + counts_T
    nb_empty = (particles == 0).sum()  # number of empty events (no particles)
    if nb_empty:
        one_hot_fix = np.zeros((len(counts_p), 3), dtype=int)
        one_hot_array = np.zeros((nb_empty, 3), dtype=int)
        one_hot_array[np.arange(nb_empty), np_rand_obj.randint(0, 3, size=(nb_empty, 1)).flatten()] = 1
        one_hot_fix[particles == 0] = one_hot_array
        counts_p += one_hot_fix[:, 0]
        counts_D += one_hot_fix[:, 1]
        counts_T += one_hot_fix[:, 2]


def gen_image(generator, args, test_set, ke, theta, phi, ini_x, ini_y, ini_z,
              exit_x=None, exit_y=None, exit_z=None, n_images=1, device="cpu"):
    """
    Generate synthetic images using a generative model.

    Args:
        generator (callable): The generative model function that takes parameters and noise
            as input and generates synthetic images.
        args (object): An object containing additional arguments and settings.
        test_set (object): An object representing the test set with source and target data ranges.
        ke (float): Kinetic energy of the particle.
        theta (float): Theta angle of the particle trajectory.
        phi (float): Phi angle of the particle trajectory.
        ini_x (float): Initial x-coordinate of the particle.
        ini_y (float): Initial y-coordinate of the particle.
        ini_z (float): Initial z-coordinate of the particle.
        exit_x (float, optional): Final x-coordinate of the particle (used for the mu case).
        exit_y (float, optional): Final y-coordinate of the particle (used for the mu case).
        exit_z (float, optional): Final z-coordinate of the particle (used for the mu case).
        n_images (int, optional): The number of synthetic images to generate.
        device (str, optional): The device on which to run the generator (e.g., "cpu" or "cuda").

    Returns:
        numpy.ndarray: An array containing the generated synthetic images, reshaped to the specified
        dimensions (n_images, 5, 5, 5).
    """
    # Kinematic parameters
    if exit_x is None:
        # p, D+, T+ case
        params = np.array([ini_x, ini_y, ini_z, ke, theta, phi])
        # Rescale
        params[:3] = np.interp(params[:3].ravel(), (test_set.min_pos, test_set.max_pos),
                               test_set.source_range).reshape(params[:3].shape)
        params[3] = np.interp(params[3], (test_set.min_ke, test_set.max_ke),
                              test_set.source_range).reshape(1)
        params[4] = np.interp(params[4], (test_set.min_theta, test_set.max_theta),
                              test_set.source_range).reshape(1)
        params[5] = np.interp(params[5], (test_set.min_phi, test_set.max_phi),
                              test_set.source_range).reshape(1)
    else:
        # mu case
        params = np.array([ini_x, ini_y, ini_z, exit_x, exit_y, exit_z, ke, theta, phi])
        # Rescale
        params[:3] = np.interp(params[:3].ravel(), (test_set.min_pos, test_set.max_pos),
                               test_set.source_range).reshape(params[:3].shape)
        params[3:6] = np.interp(params[3:6].ravel(), (test_set.min_exit_pos_mu, test_set.max_exit_pos_mu),
                                test_set.source_range).reshape(params[3:6].shape)
        params[6] = np.interp(params[6], (test_set.min_ke, test_set.max_ke),
                              test_set.source_range).reshape(1)
        params[7] = np.interp(params[7], (test_set.min_theta, test_set.max_theta),
                              test_set.source_range).reshape(1)
        params[8] = np.interp(params[8], (test_set.min_phi, test_set.max_phi),
                              test_set.source_range).reshape(1)

    # Tensors
    params = torch.tensor(np.array([params for i in range(n_images)])).float().to(device)
    noise = torch.normal(0, 1, size=(len(params), 1, args.noise_size)).to(device)  # normal noise!

    # Run the generator
    sample_image = generator(params, noise).data.cpu()

    # Rescale back
    sample_image = np.interp(sample_image.ravel(), test_set.target_range,
                             (test_set.min_charge, test_set.max_charge)).reshape(sample_image.shape)

    return sample_image.reshape(n_images, 5, 5, 5)


def centre_vertex(vtx, test_set):
    """
    Center a vertex within a cubic space defined by `test_set`. This function adjusts the vertex position to fit within
    the cubic space defined by `test_set.cube_size`. If the vertex is outside the central cube, it wraps the position by
    shifting it inside the cube and keeps track of the number of shifts in each axis using the `xyz` list.

    Parameters:
    - vtx (list of float): The vertex to be centered, represented as a list of three float values [x, y, z].
    - test_set (object): An object that defines the size of the cubic space in `cube_size`.

    Returns:
    - tuple: A tuple containing two elements:
        1. The centered vertex, represented as a list of three float values [x, y, z].
        2. A list of three integers [x_shift, y_shift, z_shift] indicating how many times the vertex has been shifted
           in each axis to center it within the cubic space.
    """
    xyz = [0, 0, 0]
    for i in range(3):
        if vtx[i] > test_set.cube_size / 2:
            vtx[i] -= test_set.cube_size
            xyz[i] += 1
        elif vtx[i] < -test_set.cube_size / 2:
            vtx[i] += test_set.cube_size
            xyz[i] -= 1
    return vtx, xyz


def eval_event_conf1(event_n, model, test_set, device="cpu"):
    """
    Evaluate an event using the given transformer model and test set (configuration 1).

    Args:
        event_n (int): The index of the event to be evaluated.
        model: The transformer neural network model for event evaluation.
        test_set: The dataset containing the events for evaluation.
        device (str, optional): The device to run the evaluation on (e.g., 'cpu' or 'cuda'). Default is 'cpu'.

    Returns:
        tuple: A tuple containing the following elements:
            - particle_images (numpy.ndarray): Rescaled particle images.
            - muon_exit (numpy.ndarray): Rescaled muon exit parameters.
            - params_true (numpy.ndarray): True event parameters.
            - params_pred (numpy.ndarray): Predicted event parameters.
            - vtx_true (float): True vertex position.
            - vtx_pred (float): Predicted vertex position.
    """
    # Evaluation mode
    model.eval()

    # Retrieve event
    event = test_set[event_n]
    muon_exit, params_true, vtx_true = event['exit_muon'], event['params'], event['ini_pos']
    event = test_set.collate_fn([event])
    src, muon, vtx_true, tgt, next_tgt, _, particle_images = event
    src = src.to(device)
    muon = muon.to(device)

    # Run encoder
    memory, vtx_pred = model.encode(src, muon, None, None)
    memory = memory.to(device)

    # Starting token
    params_pred = model.first_token.reshape(1, 1, 3)

    # Iterate over the decoder
    max_len = len(tgt)
    for i in range(max_len):
        params_out, keep_iterating_out = model.decode(params_pred, memory, None, None, None)
        out_last = params_out[-1].reshape(1, params_out.shape[1], params_out.shape[2])
        keep_iterating_last = keep_iterating_out[-1, 0]
        params_pred = torch.cat([params_pred, out_last], dim=0)

        if keep_iterating_last.argmax(0) == 0:
            break

    # Retrieve output
    params_pred = params_pred.detach().cpu().numpy()[1:, 0, :]
    vtx_true = vtx_true.cpu().numpy()
    vtx_pred = vtx_pred.detach().cpu().numpy()
    particle_images = particle_images[0]

    # Rescale to original values
    rescale(test_set, particle_images, muon_exit, params_true, params_pred, vtx_true, vtx_pred)

    return particle_images, muon_exit, params_true, params_pred, vtx_true[0], vtx_pred[0]


def eval_event_conf2(event_n, model, test_set, device="cpu"):
    """
    Evaluate an event using the given transformer model and test set (configuration 2).

    Args:
        event_n (int): The index of the event to be evaluated.
        model: The transformer neural network model for event evaluation.
        test_set: The dataset containing the events for evaluation.
        device (str, optional): The device to run the evaluation on (e.g., 'cpu' or 'cuda'). Default is 'cpu'.

    Returns:
        tuple: A tuple containing the following elements:
            - particle_images (numpy.ndarray): Rescaled particle images.
            - muon_exit (numpy.ndarray): Rescaled muon exit parameters.
            - kin_true (numpy.ndarray): True event kinematic parameters.
            - kin_pred (numpy.ndarray): Predicted event kinematic parameters.
            - pid_true (numpy.ndarray): True event particle types.
            - pid_pred (numpy.ndarray): Predicted event particle types.
            - vtx_true (float): True vertex position.
            - vtx_pred (float): Predicted vertex position.
    """
    # Evaluation mode
    model.eval()

    # Retrieve event
    event = test_set[event_n]
    muon_exit, kin_true, vtx_true = event['exit_muon'], event['params'], event['ini_pos']
    event = test_set.collate_fn([event])
    src, muon, vtx_true, tgt, next_tgt, pid_true, _, particle_images = event
    src = src.to(device)
    muon = muon.to(device)

    # Run encoder
    memory, vtx_pred = model.encode(src, muon, None, None)
    memory = memory.to(device)

    # Starting token
    kin_pred = model.first_token_kin.reshape(1, 1, 3)
    pid_pred = model.first_token_pid.reshape(1, -1)

    # Iterate over the decoder
    max_len = len(tgt)
    for i in range(max_len):
        kin_out, pid_out, keep_iterating_out = model.decode(kin_pred, pid_pred, memory, None, None, None)
        kin_last = kin_out[-1].reshape(1, kin_out.shape[1], kin_out.shape[2])
        pid_last = pid_out[-1].reshape(1, pid_out.shape[1], pid_out.shape[2])
        pid_last = torch.nn.functional.softmax(pid_last.detach(), dim=2).argmax(dim=2)
        keep_iterating_last = keep_iterating_out[-1, 0]

        # Increase size of prediction
        kin_pred = torch.cat([kin_pred, kin_last], dim=0)
        pid_pred = torch.cat([pid_pred, pid_last], dim=0)

        if keep_iterating_last.argmax(0) == 0:
            break

    # Retrieve output
    pid_true = pid_true.cpu().numpy()[:, 0]
    pid_pred = pid_pred.detach().cpu().numpy()[1:, 0]
    kin_pred = kin_pred.detach().cpu().numpy()[1:, 0, :]
    vtx_true = vtx_true.cpu().numpy()
    vtx_pred = vtx_pred.detach().cpu().numpy()
    particle_images = particle_images[0]

    # Rescale to original values
    rescale(test_set, particle_images, muon_exit, kin_true, kin_pred, vtx_true, vtx_pred)

    return particle_images, muon_exit, kin_true, kin_pred, pid_true, pid_pred, vtx_true[0], vtx_pred[0]


def rescale(test_set, particle_images, muon_exit, params_true, params_pred, vtx_true, vtx_pred):
    """
    Rescale various arrays to their original values using the provided test set's parameters.

    Args:
        test_set: The dataset containing the scaling parameters.
        particle_images (numpy.ndarray): Rescaled particle images.
        muon_exit (numpy.ndarray): Rescaled muon exit parameters.
        params_true (numpy.ndarray): True event parameters.
        params_pred (numpy.ndarray): Predicted event parameters.
        vtx_true (numpy.ndarray): True vertex position.
        vtx_pred (numpy.ndarray): Predicted vertex position.
    """
    # Rescale to original values
    for i in range(particle_images.shape[0]):
        particle_images[i] = np.interp(particle_images[i].ravel(), test_set.target_range,
                                       (test_set.min_charge, test_set.max_charge)).reshape(particle_images[i].shape)
    muon_exit[:3] = np.interp(muon_exit[:3], test_set.source_range,
                              (test_set.min_exit_pos_mu, test_set.max_exit_pos_mu)).reshape(muon_exit[:3].shape)
    muon_exit[3] = np.interp(muon_exit[3], test_set.source_range,
                             (test_set.min_ke_mu, test_set.max_ke_mu)).reshape(1)
    muon_exit[4] = np.interp(muon_exit[4], test_set.source_range,
                             (test_set.min_theta, test_set.max_theta)).reshape(1)
    muon_exit[5] = np.interp(muon_exit[5], test_set.source_range,
                             (test_set.min_phi, test_set.max_phi)).reshape(1)
    params_true[:, 0] = np.interp(params_true[:, 0].ravel(), test_set.source_range,
                                  (test_set.min_ke_p, test_set.max_ke_p)).reshape(params_true[:, 0].shape)
    params_true[:, 1] = np.interp(params_true[:, 1].ravel(), test_set.source_range,
                                  (test_set.min_theta, test_set.max_theta)).reshape(params_true[:, 1].shape)
    params_true[:, 2] = np.interp(params_true[:, 2].ravel(), test_set.source_range,
                                  (test_set.min_phi, test_set.max_phi)).reshape(params_true[:, 2].shape)
    params_pred[:, 0] = np.interp(params_pred[:, 0].ravel(), test_set.source_range,
                                  (test_set.min_ke_p, test_set.max_ke_p)).reshape(params_pred[:, 0].shape)
    params_pred[:, 1] = np.interp(params_pred[:, 1].ravel(), test_set.source_range,
                                  (test_set.min_theta, test_set.max_theta)).reshape(params_pred[:, 1].shape)
    params_pred[:, 2] = np.interp(params_pred[:, 2].ravel(), test_set.source_range,
                                  (test_set.min_phi, test_set.max_phi)).reshape(params_pred[:, 2].shape)
    vtx_true[:] = np.interp(vtx_true.ravel(), test_set.source_range,
                            (test_set.min_ini_pos, test_set.max_ini_pos)).reshape(vtx_true.shape)
    vtx_pred[:] = np.interp(vtx_pred.ravel(), test_set.source_range,
                            (test_set.min_ini_pos, test_set.max_ini_pos)).reshape(vtx_pred.shape)


def shift_position(pos, centered_pos, cube_size, test_set):
    """
    Shift a position within a given range to ensure it stays within the specified boundaries.

    Args:
        pos (float): The original position to be shifted.
        centered_pos (float): The position used for centering calculations.
        cube_size (float): The size of the cube.
        test_set (object): An object containing information about the test set.

    Returns:
        float: The shifted position, constrained to stay within the boundaries defined by
               `test_set.min_exit_pos_mu` and `test_set.max_exit_pos_mu`.
    """
    if pos not in [test_set.min_exit_pos_mu, test_set.max_exit_pos_mu]:
        return np.clip(pos - (centered_pos * cube_size // 2),
                       test_set.min_exit_pos_mu,
                       test_set.max_exit_pos_mu)
    return pos
