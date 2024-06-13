import numpy as np
from mpi4py import MPI

# ring all reduce algorithm to communicate gradients with each other
class MPITest():
    def __init__(self) -> None:
        self.layers = [{"val": np.random.randint(1, 11)} for _ in range(13)]
        self.num_devices = 4
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        
    def ring_reduce(self):
        num_layers = len(self.layers) // self.num_devices

        send_partition = self.rank
        receive_partition = (self.rank - 1 + self.num_devices) % self.num_devices
        
        # can probably optimize with sending numpy arrays later
        # scatter reduce
        for _ in range(self.num_devices - 1):
            print(f'Computer {self.rank} has list:', self.layers)
            # send functions
            send_indices = [send_partition * num_layers,
                            min(send_partition * num_layers + num_layers, len(self.layers))]
            
            sent_data = self.comm.isend(self.layers[send_indices[0]:send_indices[1]],
                                dest=(send_partition + 1) % self.num_devices)
            
            # receive functions
            receive_indices = [receive_partition * num_layers,
                            min(receive_partition * num_layers + num_layers, len(self.layers))]
            
            received_data = self.comm.irecv(source=receive_partition)
            
            received_data = received_data.wait()

            for i, layer in enumerate(self.layers[receive_indices[0]:receive_indices[1]]):
                    layer['val'] + received_data[i]['val']

            sent_data.wait() # wait on sent_data in case it's not finished yet

            # update sending and receiving partition
            send_partition = receive_partition
            receive_partition = (send_partition - 1 + self.num_devices) % self.num_devices

        # all gather
        for _ in range(self.num_devices - 1):

            # send functions
            send_indices = [send_partition * num_layers,
                            min(send_partition * num_layers + num_layers, len(self.layers))]
            
            sent_data = self.comm.isend(self.layers[send_indices[0]:send_indices[1]],
                                dest=(send_partition + 1) % self.num_devices)
            
            # receive functions
            receive_indices = [receive_partition * num_layers,
                            min(receive_partition * num_layers + num_layers, len(self.layers))]
            
            received_data = self.comm.irecv(source=receive_partition)
            
            received_data = received_data.wait()

            for i, layer in enumerate(self.layers[receive_indices[0]:receive_indices[1]]):
                    layer['val'] + received_data[i]['val']

            sent_data.wait() # wait on sent_data in case it's not finished yet

            # update sending and receiving partition
            send_partition = receive_partition
            receive_partition = (send_partition - 1 + self.num_devices) % self.num_devices

        # take the mean
        for layer in self.layers:
            layer['W'] /= self.num_devices

example = MPITest()

example.ring_reduce()