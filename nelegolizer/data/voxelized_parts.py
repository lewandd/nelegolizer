import numpy as np

stud_grid = np.array(
        [[[False,False,False,False,False],
          [False,False,False,False,False],],

        [[False, False, False,False,False],
         [False, True, True, True,False],],

        [[False,False,False,False,False],
         [False, True, True, True,False],],

        [[False,False,False,False,False],
         [False, True, True, True,False],],

        [[False,False,False,False,False],
         [False,False,False,False,False],]])

ext_empty_grid = np.array(
        [[[False, False, False,False,False,False],
          [False,False,False,False,False,False],
          [False,False,False,False,False,False],],
          
          [[False,False,False,False,False,False],
          [False,False,False,False,False,False],
          [False,False,False,False,False,False]],

          [[False,False,False,False,False,False],
          [False,False,False,False,False,False],
          [False,False,False,False,False,False]],

          [[False,False,False,False,False,False],
          [False,False,False,False,False,False],
          [False,False,False,False,False,False]],

          [[False,False,False,False,False,False],
          [False,False,False,False,False,False],
          [False,False,False,False,False,False]],

          [[False,False,False,False,False,False],
          [False,False,False,False,False,False],
          [False,False,False,False,False,False],]])

ext_stud_grid = np.array(
        [[[False,False,False,False,False,False],
          [False,False,False,False,False,False],
          [False,False,False,False,False,False],],

         [[False,False,False,False,False,False],
          [False, True, True, True,False,False],
          [False, True, True, True,False,False],],

         [[False,False,False,False,False,False],
          [False, True, True, True,False,False],
          [False, True, True, True,False,False],],

         [[False,False,False,False,False,False],
          [False, True, True, True,False,False],
          [False, True, True, True,False,False],],

         [[False,False,False,False,False,False],
          [False,False,False,False,False,False],
          [False,False,False,False,False,False],],

         [[False,False,False,False,False,False],
          [False,False,False,False,False,False],
          [False,False,False,False,False,False],]])

ext_part_grid = {
    "3005": np.array([
        [#[False,False,False,False,False,False], #
         #[False,False,False,False,False,False], #
         [False,False,False,False,False,False], #
        [ False,False,False,False,False,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False], # extended
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False], # extended
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False]],

        [#[False,False,False,False,False,False], #
         #[False,False,False,False,False,False], #
         [False, True, True, True,False,False], #
        [ False, True, True, True,False,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False], # extended
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False], # extended
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False]],

        [#[False,False,False,False,False,False], #
         #[False,False,False,False,False,False], #
         [False, True, True, True,False,False], #
         [False, True, True, True,False,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False], # extended
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False], # extended
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False]],

        [#[False,False,False,False,False,False], #
         #[False,False,False,False,False,False], #
         [False, True, True, True,False,False], #
         [False, True, True, True,False,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False], # extended
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False], # extended
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False]],

        [#[False,False,False,False,False,False], #
         #[False,False,False,False,False,False], #
         [False,False,False,False,False,False], #
         [False,False,False,False,False,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False], # extended
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False], # extended
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False]],

        # 0 extension
        [#[False,False,False,False,False,False], #
         #[False,False,False,False,False,False], #
         [False,False,False,False,False,False], # 
        [ False,False,False,False,False,False],
        [False,False,False,False,False,False],
        [ False,False,False,False,False,False],
        [ False,False,False,False,False,False], # extended
        [ False,False,False,False,False,False],
        [ False,False,False,False,False,False],
        [ False,False,False,False,False,False], # extended
        [ False,False,False,False,False,False],
        [ False,False,False,False,False,False]]]),


    "3004": np.array(
        [[#[False,False,False,False,False,False], #
          #[False,False,False,False,False,False], #
          [False,False,False,False,False,False], #
          [False,False,False,False,False,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False], # extended
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False], # extended
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False]],

        [#[False,False,False,False,False,False], #
         #[False,False,False,False,False,False], #
         [False, True, True, True,False,False], #
         [False, True, True, True,False,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False], # extended
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False], # extended
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False]],

        [#[False,False,False,False,False,False], #
         #[False,False,False,False,False,False], #
         [False, True, True, True,False,False], #
         [False, True, True, True,False,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False], # extended
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False], # extended
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False]],

        [#[False,False,False,False,False,False], #
         #[False,False,False,False,False,False], #
         [False, True, True, True,False,False], #
         [False, True, True, True,False,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False], # extended
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False], # extended
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False]],

        [#[False,False,False,False,False,False], #
         #[False,False,False,False,False,False], #
         [False,False,False,False,False,False], #
         [False,False,False,False,False,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False], # extended
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False], # extended
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False]],

        # middle extended
        [#[False,False,False,False,False,False], #
         #[False,False,False,False,False,False], #
         [False,False,False,False,False,False], #
         [False,False,False,False,False,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False], # extended
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False], # extended
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False]],

        [#[False,False,False,False,False,False], #
         #[False,False,False,False,False,False], #
         [False,False,False,False,False,False], #
         [False,False,False,False,False,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False], # extended
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False], # extended
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False]],

        [#[False,False,False,False,False,False], #
         #[False,False,False,False,False,False], #
         [False, True, True, True,False,False], #
         [False, True, True, True,False,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False], # extended
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False], # extended
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False]],

        [#[False,False,False,False,False,False], #
         #[False,False,False,False,False,False], #
         [False, True, True, True,False,False], #
         [False, True, True, True,False,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False], # extended
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False], # extended
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False]],

        [#[False,False,False,False,False,False], #
         #[False,False,False,False,False,False], #
         [False, True, True, True,False,False], #
         [False, True, True, True,False,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False], # extended
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False], # extended
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False]],

        [#[False,False,False,False,False,False], #
         #[False,False,False,False,False,False], #
         [False,False,False,False,False,False], #
         [False,False,False,False,False,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False], # extended
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False], # extended
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False]],

        [#[False,False,False,False,False,False], #
         #[False,False,False,False,False,False], #
         [False,False,False,False,False,False], #
         [False,False,False,False,False,False], # 0 extension
        [ False,False,False,False,False,False],
        [ False,False,False,False,False,False],
        [ False,False,False,False,False,False], # extended
        [ False,False,False,False,False,False],
        [ False,False,False,False,False,False],
        [ False,False,False,False,False,False], # extended
        [ False,False,False,False,False,False],
        [ False,False,False,False,False,False]]]),


    "54200": np.array(
        [[#[False,False,False,False, False, False], #
          #[False,False,False,False, False, False], #
          [False,False,False,False, False, False], #
          [False,False,False,False, False, False],
        [ False,False,False,False, False, False],
        [ False,False,False,False, False, False],
        [ False,False,False,False, False, False], # extended
        [ False,False,False,False, True, False],
        [ False,False, True, True, True, False],
        [ False,False, True, True, True, False], # extended
        [ False, True, True, True, True, False],
        [ True, True, True, True, True, False]],

        [#[False,False,False,False, False, False], #
         #[False,False,False,False, False, False], #
         [False,False,False,False, False, False], #
         [False,False,False,False, False, False],
        [ False,False,False,False, False, False],
        [ False,False,False,False, False, False],
        [ False,False,False,False, False, False], # extended
        [ False,False,False,False, True, False],
        [ False,False, True, True, True, False],
        [ False,False, True, True, True, False], # extended
        [ False, True, True, True, True, False],
        [ True, True, True, True, True, False]],

        [#[False,False,False,False, False, False], #
         #[False,False,False,False, False, False], #
         [False,False,False,False, False, False], #
         [False,False,False,False, False, False],
        [ False,False,False,False, False, False],
        [ False,False,False,False, False, False],
        [ False,False,False,False, False, False], # extended
        [False,False,False,False, True, False],
        [ False,False, True, True, True, False],
        [ False,False, True, True, True, False], # extended
        [ False, True, True, True, True, False],
        [ True, True, True, True, True, False]],

        [#[False,False,False,False, False, False], #
         #[False,False,False,False, False, False], #
         [False,False,False,False, False, False], #
         [False,False,False,False, False, False],
        [ False,False,False,False, False, False],
        [ False,False,False,False, False, False],
        [ False,False,False,False, False, False], # extended
        [False,False,False,False, True, False],
        [ False,False, True, True, True, False],
        [ False,False, True, True, True, False], # extended
        [ False, True, True, True, True, False],
        [ True, True, True, True, True, False]],

        [#[False,False,False,False, False, False], #
         #[False,False,False,False, False, False], #
         [False,False,False,False, False, False], #
         [False,False,False,False, False, False],
        [ False,False,False,False, False, False],
        [ False,False,False,False, False, False],
        [ False,False,False,False, False, False], # extended
        [False,False,False,False, True, False],
        [ False,False, True, True, True, False],
        [ False,False, True, True, True, False], # extended
        [ False, True, True, True, True, False],
        [ True, True, True, True, True, False]],
        
        [#[False,False,False,False, False, False], #
         #[False,False,False,False, False, False], #
         [False,False,False,False, False, False], #
         [False,False,False,False,False,False], # 0 extension
        [ False,False,False,False,False,False],
        [ False,False,False,False,False,False],
        [ False,False,False,False,False,False], # extended
        [ False,False,False,False,False,False],
        [ False,False,False,False,False,False],
        [ False,False,False,False,False,False], # extended
        [ False,False,False,False,False,False],
        [ False,False,False,False,False,False]]]),


    "3024": np.array(
        [[#[False,False,False,False,False,False], #
          #[False,False,False,False,False,False], #
          [False,False,False,False,False,False], #
          [False,False,False,False,False,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False]],

        [#[False,False,False,False,False,False], #
         #[False,False,False,False,False,False], #
         [False, True, True, True,False,False], #
         [False, True, True, True,False,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False]],

        [#[False,False,False,False,False,False], #
         #[False,False,False,False,False,False], #
         [False, True, True, True,False,False], #
         [False, True, True, True,False,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False]],

        [#[False,False,False,False,False,False], #
         #[False,False,False,False,False,False], #
         [False, True, True, True,False,False], #
         [False, True, True, True,False,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False]],

        [#[False,False,False,False,False,False], #
         #[False,False,False,False,False,False], #
         [False,False,False,False,False,False], #
         [False,False,False,False,False,False],
        [ True, True, True, True, True,False],
        [ True, True, True, True, True,False]],

        [#[False,False,False,False,False,False], #
         #[False,False,False,False,False,False], #
         [False,False,False,False,False,False], #
         [False,False,False,False,False,False],  # 0 xtension
        [False,False,False,False,False,False],
        [False,False,False,False,False,False]]])
}

part3004_rot_90 = np.array( 
    #                  vertical extended fill at 6th position                   
    [[#[False, False, False, False, False, False, False, False, False, False, False, False], #
      #[False, False, False, False, False, False, False, False, False, False, False, False], #
      [False, False, False, False, False, False, False, False, False, False, False, False], #
      [False, False, False, False, False, False, False, False, False, False, False, False],
    [ True, True, True, True, True, True, True, True, True, True, True, False],
    [ True, True, True, True, True, True, True, True, True, True, True, False],
    [ True, True, True, True, True, True, True, True, True, True, True, False],# extended fill
    [ True, True, True, True, True, True, True, True, True, True, True, False],
    [ True, True, True, True, True, True, True, True, True, True, True, False],
    [ True, True, True, True, True, True, True, True, True, True, True, False], # extended fill
    [ True, True, True, True, True, True, True, True, True, True, True, False],
    [ True, True, True, True, True, True, True, True, True, True, True, False]],

    [#[False, False, False, False, False, False, False, False, False, False, False, False], #
     #[False, False, False, False, False, False, False, False, False, False, False, False], #
     [False, True, True, True, False, False, False, True, True, True, False, False], #
     [False, True, True, True, False, False, False, True, True, True, False, False],
    [ True, True, True, True, True, True, True, True, True, True, True, False],
    [ True, True, True, True, True, True, True, True, True, True, True, False], 
    [ True, True, True, True, True, True, True, True, True, True, True, False],# extended fill
    [ True, True, True, True, True, True, True, True, True, True, True, False],
    [ True, True, True, True, True, True, True, True, True, True, True, False],
    [ True, True, True, True, True, True, True, True, True, True, True, False], # extended fill
    [ True, True, True, True, True, True, True, True, True, True, True, False],
    [ True, True, True, True, True, True, True, True, True, True, True, False]],

    [#[False, False, False, False, False, False, False, False, False, False, False, False], #
     #[False, False, False, False, False, False, False, False, False, False, False, False], #
     [False, True, True, True, False, False, False, True, True, True, False, False], #
     [False, True, True, True, False, False, False, True, True, True, False, False],
    [ True, True, True, True, True, True, True, True, True, True, True, False],
    [ True, True, True, True, True, True, True, True, True, True, True, False],
    [ True, True, True, True, True, True, True, True, True, True, True, False], # extended fill
    [ True, True, True, True, True, True, True, True, True, True, True, False],
    [ True, True, True, True, True, True, True, True, True, True, True, False],
    [ True, True, True, True, True, True, True, True, True, True, True, False], # extended fill
    [ True, True, True, True, True, True, True, True, True, True, True, False],
    [ True, True, True, True, True, True, True, True, True, True, True, False]],

    [#[False, False, False, False, False, False, False, False, False, False, False, False], #
     #[False, False, False, False, False, False, False, False, False, False, False, False], #
     [False, True, True, True, False, False, False, True, True, True, False, False], #
     [False, True, True, True, False, False, False, True, True, True, False, False],
    [ True, True, True, True, True, True, True, True, True, True, True, False],
    [ True, True, True, True, True, True, True, True, True, True, True, False],
    [ True, True, True, True, True, True, True, True, True, True, True, False], # extended fill
    [ True, True, True, True, True, True, True, True, True, True, True, False],
    [ True, True, True, True, True, True, True, True, True, True, True, False],
    [ True, True, True, True, True, True, True, True, True, True, True, False], # extended fill
    [ True, True, True, True, True, True, True, True, True, True, True, False],
    [ True, True, True, True, True, True, True, True, True, True, True, False]],

    [#[False, False, False, False, False, False, False, False, False, False, False, False], #
     #[False, False, False, False, False, False, False, False, False, False, False, False], #
     [False, False, False, False, False, False, False, False, False, False, False, False], #
     [False, False, False, False, False, False, False, False, False, False, False, False],
    [ True, True, True, True, True, True, True, True, True, True, True, False],
    [ True, True, True, True, True, True, True, True, True, True, True, False],
    [ True, True, True, True, True, True, True, True, True, True, True, False], # extended fill
    [ True, True, True, True, True, True, True, True, True, True, True, False],
    [ True, True, True, True, True, True, True, True, True, True, True, False],
    [ True, True, True, True, True, True, True, True, True, True, True, False], # extended fill
    [ True, True, True, True, True, True, True, True, True, True, True, False],
    [ True, True, True, True, True, True, True, True, True, True, True, False]],
    
    [#[False, False, False, False, False, False, False, False, False, False, False, False], #
     #[False, False, False, False, False, False, False, False, False, False, False, False], #
     [False, False, False, False, False, False, False, False, False, False, False, False], #
     [False, False, False, False, False, False, False, False, False, False, False, False], # extended 0
    [False, False, False, False, False, False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False, False, False, False, False, False], # extended fill
    [False, False, False, False, False, False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False, False, False, False, False, False], # extended fill
    [False, False, False, False, False, False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False, False, False, False, False, False]]])

part_54200_rot_90 = np.array([[
    #[False, False, False, False, False, False], #
    #[False, False, False, False, False, False], #
    [False, False, False, False, False, False], #
    [False, False, False, False, False, False],
    [False, False, False, False, False, False],
    [False, False, False, False, False, False],
    [False, False, False, False, False, False], # extended fill
    [False, False, False, False, False, False],
    [False, False, False, False, False, False],
    [False, False, False, False, False, False], # extended fill
    [False, False, False, False, False, False],
    [ True, True, True, True, True, False]],

    [#[False, False, False, False, False, False], #
     #[False, False, False, False, False, False], #
     [False, False, False, False, False, False], #
     [False, False, False, False, False, False],
    [False, False, False, False, False, False],
    [False, False, False, False, False, False],
    [False, False, False, False, False, False], # extended fill
    [False, False, False, False, False, False],
    [False, False, False, False, False, False],
    [False, False, False, False, False, False], # extended fill
    [ True, True, True, True, True, False],
    [ True, True, True, True, True, False]],

    [#[False, False, False, False, False, False], #
     #[False, False, False, False, False, False], #
     [False, False, False, False, False, False], #
     [False, False, False, False, False, False],
    [False, False, False, False, False, False],
    [False, False, False, False, False, False],
    [False, False, False, False, False, False], # extended fill
    [False, False, False, False, False, False],
    [ True, True, True, True, True, False],
    [ True, True, True, True, True, False], # extended fill
    [ True, True, True, True, True, False],
    [ True, True, True, True, True, False]],

    [#[False, False, False, False, False, False], #
     #[False, False, False, False, False, False], #
     [False, False, False, False, False, False], #
     [False, False, False, False, False, False],
    [False, False, False, False, False, False],
    [False, False, False, False, False, False],
    [False, False, False, False, False, False], # extended fill
    [False, False, False, False, False, False],
    [ True, True, True, True, True, False],
    [ True, True, True, True, True, False], # extended fill
    [ True, True, True, True, True, False],
    [ True, True, True, True, True, False]],

    [#[False, False, False, False, False, False], #
     #[False, False, False, False, False, False], #
     [False, False, False, False, False, False], #
     [False, False, False, False, False, False],
    [False, False, False, False, False, False],
    [False, False, False, False, False, False],
    [False, False, False, False, False, False], # extended fill
    [ True, True, True, True, True, False],
    [ True, True, True, True, True, False],
    [ True, True, True, True, True, False], # extended fill
    [ True, True, True, True, True, False],
    [ True, True, True, True, True, False]],
    
    [#[False, False, False, False, False, False], #
     #[False, False, False, False, False, False], #
     [False, False, False, False, False, False], #
     [False, False, False, False, False, False],
     [False, False, False, False, False, False],
     [False, False, False, False, False, False],
     [False, False, False, False, False, False], # extended fill
     [False, False, False, False, False, False],
     [False, False, False, False, False, False],
     [False, False, False, False, False, False], # extended fill
     [False, False, False, False, False, False],
     [False, False, False, False, False, False]]])

part_54200_rot_180 = np.array([[
    #[False, False, False, False, False, False], #
    #[False, False, False, False, False, False], #
    [False, False, False, False, False, False], #
    [False, False, False, False, False, False],
    [False, False, False, False, False, False],
    [False, False, False, False, False, False],
    [False, False, False, False, False, False], # extended
    [ True, False, False, False, False, False],
    [ True, True, True, False, False, False],
    [ True, True, True, False, False, False], #
    [ True, True, True, True, False, False],
    [ True, True, True, True, True, False]],

    [#[False, False, False, False, False, False], #
     #[False, False, False, False, False, False], #
     [False, False, False, False, False, False], #
     [False, False, False, False, False, False],
    [False, False, False, False, False, False],
    [False, False, False, False, False, False],
    [False, False, False, False, False, False], #
    [ True, False, False, False, False, False],
    [ True, True, True, False, False, False],
    [ True, True, True, False, False, False], #
    [ True, True, True, True, False, False],
    [ True, True, True, True, True, False]],

    [#[False, False, False, False, False, False], #
     #[False, False, False, False, False, False], #
     [False, False, False, False, False, False], #
     [False, False, False, False, False, False],
    [False, False, False, False, False, False],
    [False, False, False, False, False, False],
    [False, False, False, False, False, False], #
    [ True, False, False, False, False, False],
    [ True, True, True, False, False, False],
    [ True, True, True, False, False, False], #
    [ True, True, True, True, False, False],
    [ True, True, True, True, True, False]],

    [#[False, False, False, False, False, False], #
     #[False, False, False, False, False, False], #
     [False, False, False, False, False, False], #
     [False, False, False, False, False, False],
    [False, False, False, False, False, False],
    [False, False, False, False, False, False],
    [False, False, False, False, False, False], #
    [ True, False, False, False, False, False],
    [ True, True, True, False, False, False],
    [ True, True, True, False, False, False],#
    [ True, True, True, True, False, False],
    [ True, True, True, True, True, False]],

    [#[False, False, False, False, False, False], #
     #[False, False, False, False, False, False], #
     [False, False, False, False, False, False], #
     [False, False, False, False, False, False],
    [False, False, False, False, False, False],
    [False, False, False, False, False, False],
    [False, False, False, False, False, False], #
    [ True, False, False, False, False, False],
    [ True, True, True, False, False, False],
    [ True, True, True, False, False, False], #
    [ True, True, True, True, False, False],
    [ True, True, True, True, True, False]],

    [#[False, False, False, False, False, False], #
     #[False, False, False, False, False, False], #
     [False, False, False, False, False, False], #
     [False, False, False, False, False, False],
    [False, False, False, False, False, False],
    [False, False, False, False, False, False],
    [False, False, False, False, False, False], #
    [False, False, False, False, False, False],
    [False, False, False, False, False, False],
    [False, False, False, False, False, False], #
    [False, False, False, False, False, False],
    [False, False, False, False, False, False]]])

part_54200_rot_270 = np.array([[
    #[False, False, False, False, False, False], #
    #[False, False, False, False, False, False], #
    [False, False, False, False, False, False], #
    [False, False, False, False, False, False],
    [False, False, False, False, False, False],
    [False, False, False, False, False, False],
    [False, False, False, False, False, False], #
    [ True, True, True, True, True, False],
    [ True, True, True, True, True, False],
    [ True, True, True, True, True, False], #
    [ True, True, True, True, True, False],
    [ True, True, True, True, True, False]],

    [#[False, False, False, False, False, False], #
     #[False, False, False, False, False, False], #
     [False, False, False, False, False, False], #
     [False, False, False, False, False, False],
    [False, False, False, False, False, False],
    [False, False, False, False, False, False],
    [False, False, False, False, False, False], #
    [False, False, False, False, False, False],
    [ True, True, True, True, True, False],
    [ True, True, True, True, True, False], #
    [ True, True, True, True, True, False],
    [ True, True, True, True, True, False]],

    [#[False, False, False, False, False, False], #
     #[False, False, False, False, False, False], #
     [False, False, False, False, False, False], #
     [False, False, False, False, False, False],
    [False, False, False, False, False, False],
    [False, False, False, False, False, False],
    [False, False, False, False, False, False], #
    [False, False, False, False, False, False],
    [ True, True, True, True, True, False],
    [ True, True, True, True, True, False], #
    [ True, True, True, True, True, False],
    [ True, True, True, True, True, False]],

    [#[False, False, False, False, False, False], #
     #[False, False, False, False, False, False], #
     [False, False, False, False, False, False], #
     [False, False, False, False, False, False],
    [False, False, False, False, False, False],
    [False, False, False, False, False, False],
    [False, False, False, False, False, False], #
    [False, False, False, False, False, False],
    [False, False, False, False, False, False],
    [False, False, False, False, False, False], #
    [ True, True, True, True, True, False],
    [ True, True, True, True, True, False]],

    [#[False, False, False, False, False, False], #
     #[False, False, False, False, False, False], #
     [False, False, False, False, False, False], #
     [False, False, False, False, False, False],
    [False, False, False, False, False, False],
    [False, False, False, False, False, False],
    [False, False, False, False, False, False], #
    [False, False, False, False, False, False],
    [False, False, False, False, False, False],
    [False, False, False, False, False, False], #
    [False, False, False, False, False, False],
    [ True, True, True, True, True, False]],

    [#[False, False, False, False, False, False], #
     #[False, False, False, False, False, False], #
     [False, False, False, False, False, False], #
     [False, False, False, False, False, False],
    [False, False, False, False, False, False],
    [False, False, False, False, False, False],
    [False, False, False, False, False, False], #
    [False, False, False, False, False, False],
    [False, False, False, False, False, False],
    [False, False, False, False, False, False], #
    [False, False, False, False, False, False],
    [False, False, False, False, False, False]]])

ext_part_grid2 = {
    "3005": {0: ext_part_grid["3005"],
             90: ext_part_grid["3005"],
             180: ext_part_grid["3005"],
             270: ext_part_grid["3005"]},
    "3004": {0: ext_part_grid["3004"],
             90: part3004_rot_90,
             180: ext_part_grid["3004"],
             270: part3004_rot_90},
    "3024": {0: ext_part_grid["3024"],
             90: ext_part_grid["3024"],
             180: ext_part_grid["3024"],
             270: ext_part_grid["3024"]},
    "54200": {0: ext_part_grid["54200"],
              90: part_54200_rot_90,
              180: part_54200_rot_180,
              270: part_54200_rot_270}
    }

#ext_part_grid_rot90 = {
#    "3005": ext_part_grid_rot0["3005"],
#
#
#    "3004":
#}

part_grid = {
    "3005": np.array([
        [[False,False,False,False,False],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True]],

        [[False, True, True, True,False],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True]],

        [[False, True, True, True,False],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True]],

        [[False, True, True, True,False],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True]],

        [[False,False,False,False,False],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True]]]),


    "3004": np.array(
        [[[False,False,False,False,False],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True]],

        [[False, True, True, True,False],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True]],

        [[False, True, True, True,False],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True]],

        [[False, True, True, True,False],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True]],

        [[False,False,False,False,False],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True]],

        [[False,False,False,False,False],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True]],

        [[False, True, True, True,False],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True]],

        [[False, True, True, True,False],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True]],

        [[False, True, True, True,False],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True]],

        [[False,False,False,False,False],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True],
        [ True, True, True, True, True]]]),


    "54200": np.array(
        [[[False,False,False,False, False],
        [ False,False,False,False, False],
        [ False,False,False,False, False],
        [ False,False,False,False, True],
        [ False,False, True, True, True],
        [ False, True, True, True, True],
        [ True, True, True, True, True]],

        [[False,False,False,False, False],
        [ False,False,False,False, False],
        [ False,False,False,False, False],
        [False,False,False,False, True],
        [ False,False, True, True, True],
        [ False, True, True, True, True],
        [ True, True, True, True, True]],

        [[False,False,False,False, False],
        [ False,False,False,False, False],
        [ False,False,False,False, False],
        [False,False,False,False, True],
        [ False,False, True, True, True],
        [ False, True, True, True, True],
        [ True, True, True, True, True]],

        [[False,False,False,False, False],
        [ False,False,False,False, False],
        [ False,False,False,False, False],
        [False,False,False,False, True],
        [ False,False, True, True, True],
        [ False, True, True, True, True],
        [ True, True, True, True, True]],

        [[False,False,False,False, False],
        [ False,False,False,False, False],
        [ False,False,False,False, False],
        [False,False,False,False, True],
        [ False,False, True, True, True],
        [ False, True, True, True, True],
        [ True, True, True, True, True]]]),


    "3024": np.array(
        [[[False,False,False,False,False],
        [ True, True, True, True, True],
        [ True, True, True, True, True]],

        [[False, True, True, True,False],
        [ True, True, True, True, True],
        [ True, True, True, True, True]],

        [[False, True, True, True,False],
        [ True, True, True, True, True],
        [ True, True, True, True, True]],

        [[False, True, True, True,False],
        [ True, True, True, True, True],
        [ True, True, True, True, True]],

        [[False,False,False,False,False],
        [ True, True, True, True, True],
        [ True, True, True, True, True]]])
}