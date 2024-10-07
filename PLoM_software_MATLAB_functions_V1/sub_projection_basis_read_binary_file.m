function [MatRg_mDP] = sub_projection_basis_read_binary_file(filename,n_d,mDP)

   %  Copyright: Christian Soize, Universite Gustave Eiffel, 02 June 2024
   %
   %  Software     : Probabilistic Learning on Manifolds (PLoM) 
   %  Function name: sub_projection_basis_read_binary_file.m
   %  Subject      : read nd,mDP,MatRg_mDP(nd,mDP) on a Binary File  where nd should be equal to n_d and mDP <= n_d,  
   
   %--- INPUTS
   %          filename           : file name of the type fileName = 'data.bin'
   %          n_d                : dimension of the realizations in the training dataset
   %          mDP                : maximum number of the projection basis vectors that are read on a binary file
   %
   %--- OUTPUTS
   %
   %          MatRg_mDP(nd,mDP)  : mDP vectors of the projection basis

   %--- Open the file in binary read mode
   fileID = fopen(filename, 'r');               % file name must be of the type fileName = 'data.bin';
   
   %--- Check that the file is correctly opened   
   if fileID == -1
      error('STOP1 in sub_projection_basis_read_binary_file: impossible to open the file %s', filename);
   end

   %--- Read nd and mDPtemp
   nd      = fread(fileID, 1, 'int');
   mDPtemp = fread(fileID, 1, 'int');
   
   %--- Checking data
   if nd ~= n_d
      error('STOP2 in sub_projection_basis_read_binary_file: the read dimension, nd, must be equal to n_d')
   end
   if mDPtemp ~= mDP
      error('STOP3 in sub_projection_basis_read_binary_file: the read dimension, mDP, is not coherent with the given value of mDP ')
   end

   % Read MatRg_mDP(nd,mDP) 
   MatRg_mDP = fread(fileID, [nd, mDPtemp], 'double');
   
   % Close the file
   fclose(fileID);

   return
end



