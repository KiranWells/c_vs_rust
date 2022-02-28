// code adapted from:
// https://stackoverflow.com/questions/2654480/writing-bmp-image-in-pure-c-c-without-other-libraries
// used as a bitmap saving library

use std::{fs, io::Write};

/// red, green, & blue
pub const BYTES_PER_PIXEL: usize = 3;
const FILE_HEADER_SIZE: usize = 14;
const INFO_HEADER_SIZE: usize = 40;

pub fn generate_bitmap_image(
    image: &[u8],
    height: usize,
    width: usize,
    filename: &str,
) -> Result<(), std::io::Error> {
    let width_in_bytes = width * BYTES_PER_PIXEL;

    let padding = [0, 0, 0];
    let padding_size = (4 - (width_in_bytes) % 4) % 4;
    let padding = padding.split_at(padding_size).0;

    let stride = (width_in_bytes) + padding_size;

    let mut image_file = fs::File::options()
        .write(true)
        .truncate(true)
        .create(true)
        .open(filename)?;

    let file_header = create_bitmap_file_header(height, stride);
    image_file.write(&file_header)?;

    let info_header = create_bitmap_info_header(height, width);
    image_file.write(&info_header)?;

    for chunk in image.chunks(width_in_bytes) {
        image_file.write(&chunk)?;
        image_file.write(&padding)?;
    }
    Ok(())
}

fn create_bitmap_file_header(height: usize, stride: usize) -> [u8; 14] {
    let file_size = FILE_HEADER_SIZE + INFO_HEADER_SIZE + (stride * height);

    let mut file_header = [0; 14];
    // [
    //     0,0,     /// signature
    //     0,0,0,0, /// image file size in bytes
    //     0,0,0,0, /// reserved
    //     0,0,0,0, /// start of pixel array
    // ];

    file_header[0] = 'B' as u8;
    file_header[1] = 'M' as u8;
    file_header[2] = (file_size) as u8;
    file_header[3] = (file_size >> 8) as u8;
    file_header[4] = (file_size >> 16) as u8;
    file_header[5] = (file_size >> 24) as u8;
    file_header[10] = (FILE_HEADER_SIZE + INFO_HEADER_SIZE) as u8;

    return file_header;
}

fn create_bitmap_info_header(height: usize, width: usize) -> [u8; 40] {
    let mut info_header = [0; 40];
    // [
    //     0,0,0,0, /// header size
    //     0,0,0,0, /// image width
    //     0,0,0,0, /// image height
    //     0,0,     /// number of color planes
    //     0,0,     /// bits per pixel
    //     0,0,0,0, /// compression
    //     0,0,0,0, /// image size
    //     0,0,0,0, /// horizontal resolution
    //     0,0,0,0, /// vertical resolution
    //     0,0,0,0, /// colors in color table
    //     0,0,0,0, /// important color count
    // ];

    info_header[0] = (INFO_HEADER_SIZE) as u8;
    info_header[4] = (width) as u8;
    info_header[5] = (width >> 8) as u8;
    info_header[6] = (width >> 16) as u8;
    info_header[7] = (width >> 24) as u8;
    info_header[8] = (height) as u8;
    info_header[9] = (height >> 8) as u8;
    info_header[10] = (height >> 16) as u8;
    info_header[11] = (height >> 24) as u8;
    info_header[12] = (1) as u8;
    info_header[14] = (BYTES_PER_PIXEL * 8) as u8;

    return info_header;
}
