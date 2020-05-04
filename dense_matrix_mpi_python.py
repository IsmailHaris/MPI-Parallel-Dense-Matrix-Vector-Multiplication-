#coding: utf-8 
#command mpirun -n nb_procs python python_file.py
# Import mpi4py and call MPI
import mpi4py
import mpi4py.MPI as MPI
import time

# Initialise the environnement MPI : automatically done when calling import mpi4py
print(MPI.Is_initialized())
#mpi4py.rc.initialize = False
#MPI.Init()

"""
----------------------------------------------------------------------------
Communicator : by default MPI.COMM_WORLD gather all the procs.
functions : Get_rank(), Get_size(), Get_processor_name().
Communications exchange messages between a sender and a receiver. They can 
be blocking or non-blocking, point to point, global, or one-sided.
Messages contains communicator, id of sender, id of receiver, a tag, message
content (data exchanged, type of data, size of data)
Logic of the program : Master Slaves configuration. 
I will mainly use comm.Send and comm.Recv since sending numpy arrays is faster. 


----------------------------------------------------------------------------
"""
import numpy as np
"""============================
Size Matrix : nrows x nrows 
Size vector : nrows x 1
============================="""

nrows = 10000

comm = MPI.COMM_WORLD
nb_procs = comm.Get_size()
rank = comm.Get_rank()


#We have to define a partition function to define local nrows
def partition(rank, size, N):
    n = N//size + ((N%size)>rank-1)
    s = (rank-1)*(N//size)
    if (N%size)>rank-1:
        s = s+rank-1
    else:
        s = s+N%size
    return s, s+n-1


matrix = np.empty([nrows, nrows],dtype=np.float64)
y = np.empty([nrows,1], dtype = np.float64)
vector = np.empty([nrows, 1], dtype = np.float64)



"""=================================
Master : Proc 0 
Tasks : 
    *Generate Matrix 
    *Compute local nrows
    *Distribute the Matrix to slaves
    (Send local matrices)
==================================="""


if rank == 0:
    print ("Running %d parallel MPI processes" % nb_procs)

    # Generate Matrix  : 
    matrix = np.random.rand(nrows, nrows).astype(np.float64)

    #Partition the Matrix and store the partitions at respective slaves
    #The proc 0 does not work( my choice), we distribute the calculations among the slaves
    for i in range(1, nb_procs):
        
        local_start_index, local_end_index = partition(i, nb_procs-1, nrows)

        #Compute and Send local nrows
        local_nrows = np.array([local_end_index+1-local_start_index],'i')
        comm.Send([local_nrows,MPI.INT], dest = i, tag = 0)
        
        #Compute and Send the local matrices to the slaves
        #print(matrix[local_start_index:local_end_index+1].shape)
        local_matrix = np.asarray(matrix[local_start_index:local_end_index+1])
        comm.Send([local_matrix,MPI.DOUBLE], dest = i, tag = 1)
        

#"""==================================================
#Slaves 
#Tasks :
#    * Create local nrows and receive it from master
#    * Create local matrix and receive it from master
#======================================================"""

if (rank>=1 and rank <=nb_procs-1) :
    print ("I am proc {} and I compute the {} partition".format(rank, rank-1)) 
    
    #create and receive the local nrows
    local_nrows = np.array([1],'i')
    comm.Recv([local_nrows,MPI.INT], source = 0, tag=0)  
    #print("proc {} received local_nrows".format(rank))
    #print(local_nrows)
    
    #create and receive the local matrix
    nrows_local = local_nrows[0]
    local_matrix = np.empty([nrows_local, nrows], dtype=np.float64)
    comm.Recv([local_matrix,MPI.DOUBLE], source = 0, tag=1)
    #print("proc {} received local_matrix".format(rank))
    #print(local_matrix)


"""============================================
Broadcast the vector from master to slaves
=============================================="""   
    
if rank == 0:
    vector = np.asarray(np.random.rand(nrows,1))
else:
    vector = np.empty([nrows,1], dtype = np.float64)
comm.Bcast(vector, root=0)


"""=============================================
The Slaves computes the local_y and
send it to master
============================================"""

if (rank>=1 and rank<=nb_procs-1):
    #print("proc {} received vector".format(rank))
    
    # compute the local result at workers
    #print("proc {} starts computation".format(rank))
    #Time each proc compute the Matrix-Vector Multiplication
    mpi_start =time.time()
    local_y = np.asarray(local_matrix.dot(vector)).astype(np.float64)
    mpi_end =time.time()
    #print("proc {} finished computation".format(rank))
    
    #Send y_local to master
    comm.Send([local_y,MPI.DOUBLE], dest = 0, tag=2)
    #print("proc{} send local_y".format(rank))
    #Send time of computation 
    local_mpi_time = np.array([mpi_end-mpi_start], dtype=np.float64)
    comm.Send([local_mpi_time, MPI.DOUBLE], dest =0, tag =3)

"""===========================================
Master receive the y_local and gather all in y
+ Check the results by computing the norm
==========================================="""
if rank ==0:
    total_mpi_time = 0
    max_mpi_time_proc = 0
    for i in range(1,nb_procs):
        local_start_index, local_end_index = partition(i, nb_procs-1, nrows)
        local_y = np.empty([local_end_index+1-local_start_index,1],dtype=np.float64)
        comm.Recv([local_y, MPI.DOUBLE],source = i, tag = 2)
        #print("master received local_y from proc {}".format(i))
        local_mpi_time = np.empty([1], dtype=np.float64)
        comm.Recv([local_mpi_time, MPI.DOUBLE], source =i , tag =3)
        y[local_start_index:local_end_index+1] = local_y
        #the total time of all procs to compute their partitions of the Matrix Vector Multiplication
        total_mpi_time += local_mpi_time
        #the maximum time used by a single proc
        if local_mpi_time>max_mpi_time_proc:
            max_mpi_time_proc = local_mpi_time
        average_mpi_time = total_mpi_time/(nb_procs-1)
    
    start_normal = time.time()
    y_normal = np.linalg.norm(matrix.dot(vector))
    end_normal = time.time()
    #Check results : Frobenius norm of the difference
    print("Sequential norm :", y_normal, "Sequential time :", end_normal-start_normal)
    print("MPI norm :", np.linalg.norm(y))
    print("MPI total time of computation of all procs :", total_mpi_time[0])
    print("MPI maximum time used by a single proc for computation :", max_mpi_time_proc[0])
    print("MPI average time used by a single proc for computation :", average_mpi_time[0])

# Close the environnement MPI (automatically done)
#print (MPI.Is_finalized())
#mpi4py.rc.finalize = False
#MPI.Finalize()

