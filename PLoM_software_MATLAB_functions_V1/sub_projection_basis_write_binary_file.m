function [] = sub_projection_basis_write_binary_file(filename,n_d,mDP,MatRg_mDP)

   %  Copyright: Christian Soize, Universite Gustave Eiffel, 02 June 2024
   %
   %  Software     : Probabilistic Learning on Manifolds (PLoM) 
   %  Function name: sub_projection_basis_write_binary_file.m
   %  Subject      : write n_d,mDP,MatRg_mDP(n_d,mDP) on a Binary File  filename of type 'data.bin'  

   % Open the file in binary write mode
   fileID = fopen(filename,'w');
   
   %--- Check that the file is correctly opened   
   if fileID == -1
      error('STOP1 in sub_projection_basis_write_binary_file: impossible to open the file %s', filename);
   end
   
   % Write n_d and mDP
   fwrite(fileID,n_d,'int');
   fwrite(fileID,mDP,'int');
   
   % Write MatRg_mDP(n_d,mDP)
   fwrite(fileID,MatRg_mDP,'double');
   
   % Close the file
   fclose(fileID);
   return
end
