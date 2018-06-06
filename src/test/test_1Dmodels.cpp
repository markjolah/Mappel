/** @file test_models.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018
 * @brief tests for basic PointEmitterModel properties
 */

#include "test_mappel.h"
#include "Mappel/Gauss1DMLE.h"
#include "Mappel/Gauss1DMAP.h"

using TypesModel1D = ::testing::Types<mappel::Gauss1DMLE,mappel::Gauss1DMAP> ;
TYPED_TEST_CASE(TestModel1D, TypesModel1D);


TYPED_TEST(TestModel1D, num_dim) {
    EXPECT_EQ(this->model.num_dim,1)<<"1D Model";
}
