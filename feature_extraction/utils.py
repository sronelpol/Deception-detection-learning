import subprocess


def run_open_face_in_docker(host_input_file_path, host_output_file_path,
                            container_name="distracted_knuth", mode="-gaze",):
    # Copy the input file from the host to the container
    input_file_name = host_input_file_path.split("/")[-1]
    output_file_name = host_output_file_path.split("/")[-1]
    command = ["docker", "cp", host_input_file_path, f"{container_name}:/home/openface-build"]
    subprocess.check_output(command)

    # Define the OpenFace command to be run
    openface_path = "build/bin/FeatureExtraction"
    openface_command = [openface_path, "-f", input_file_name, "-of", output_file_name, mode]

    # Run the OpenFace command
    command = ["docker", "exec", container_name] + openface_command
    subprocess.check_output(command)

    # Copy the output file from the container to the host
    command = ["docker", "cp", f"{container_name}:{output_file_name}", host_output_file_path]
    subprocess.check_output(command)
