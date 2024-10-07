function [MatRg_mDP] = sub_projection_basis_read_text_file(filename,n_d,mDP)

   %  Copyright: Christian Soize, Universite Gustave Eiffel, 02 June 2024
   %
   %  Software     : Probabilistic Learning on Manifolds (PLoM) 
   %  Function name: sub_projection_basis_read_text_file.m
   %  Subject      : read nd,mDP,MatRg_mDP(n_d,mDP) on a Text File  where n_d should be equal to n_d and mDP <= n_d,  
   
   %--- INPUTS
   %          filename           : file name of the type fileName = 'data.txt'
   %          n_d                : dimension of the realizations in the training dataset
   %          mDP                : maximum number of the projection basis vectors that are read on a binary file
   %
   %--- OUTPUTS
   %
   %          MatRg_mDP(nd,mDP)  : mDP vectors of the projection basis

   %--- Open the file in text read mode
   fileID = fopen(filename, 'r');               % file name must be of the type fileName = 'data.txt';
   
   %--- Check that the file is correctly opened   
   if fileID == -1
      error('STOP1 in sub_ISDE_projection_basis_read_binary_file: impossible to open the file %s', filename);
   end

   %--- Read nd and mDPtemp
   nd      = str2double(fgetl(fileID));
   mDPtemp = str2double(fgetl(fileID));

   %--- Checking data
   if nd ~= n_d
      error('STOP2 in sub_ISDE_projection_basis_read_text_file: the read dimension, nd, must be equal to n_d')
   end
   if mDPtemp ~= mDP
      error('STOP3 in sub_ISDE_projection_basis_read_text_file: the read dimension, mDP, is not coherent with the given value of mDP ')
   end

   %--- Initialize MatRg_mDP
   MatRg_mDP = zeros(n_d,mDP);

   %--- Read MatRg_mDP
   for i = 1:n_d
       line = fgetl(fileID);
       MatRg_mDP(i, :) = sscanf(line, '%f');
   end

   % Close the file
   fclose(fileID);

   return
end


