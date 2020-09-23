/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
package nyway.nd4j.samples.image.transform

import org.bytedeco.javacv.Java2DFrameConverter
import org.bytedeco.javacv.OpenCVFrameConverter.ToIplImage
import org.bytedeco.opencv.global.opencv_highgui.cvShowImage
import org.bytedeco.opencv.global.opencv_highgui.cvWaitKey
import org.bytedeco.opencv.opencv_core.IplImage
import org.bytedeco.opencv.opencv_core.Scalar
import org.datavec.image.data.ImageWritable
import org.datavec.image.loader.Java2DNativeImageLoader
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.transform.*
import java.awt.image.BufferedImage
import java.util.*


object TestTransformation {

    @JvmStatic
    fun main(args: Array<String>) {
        // file => matrix
        val imageLoader = NativeImageLoader()
        val writable = imageLoader.asWritable("/run/media/fabien/TOSHIBA/IA/ISIC/2019/HAM10000_images_part_2/ISIC_0029306.jpg")

//        val transform: ImageTransform = BoxImageTransform(Random(123), 800, 600).borderValue(Scalar.GRAY)
//        val transform: ImageTransform = EqualizeHistTransform()
//        val transform: ImageTransform = RotateImageTransform(Random(123), 45F)
        val transform: ImageTransform = FlipImageTransform(0)
//        val transform: ImageTransform = WarpImageTransform(Random(123), 180F)
        //val transform: ImageTransform = LargestBlobCropTransform(Random(123))
//        val transform: ImageTransform =  ColorConversionTransform()
//        val transform: ImageTransform =  CropImageTransform(50)

        val transformed = transform.transform(writable)
        val matrix = imageLoader.asImageMatrix("/run/media/fabien/TOSHIBA/IA/ISIC/2019/HAM10000_images_part_2/ISIC_0029306.jpg")
//        println(matrix.image)

//        // matrix => image + display
//        val asBufferedImage = Java2DNativeImageLoader().asBufferedImage(matrix.image)
//        cvShowImage("MyImage", toIplImage(asBufferedImage))
//        cvWaitKey()

        val transformedMatrix = imageLoader.asMatrix(transformed)
        // matrix => image + display
        val asBufferedImage2 = Java2DNativeImageLoader().asBufferedImage(transformedMatrix)
        cvShowImage("MyImage", toIplImage(asBufferedImage2))
        cvWaitKey()
    }

    fun toIplImage(bufImage: BufferedImage?): IplImage? {
        val iplConverter = ToIplImage()
        val java2dConverter = Java2DFrameConverter()
        return iplConverter.convert(java2dConverter.convert(bufImage))
    }

}



