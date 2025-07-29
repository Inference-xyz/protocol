// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Test.sol";
import "../contracts/ContestFactory.sol";
import "../contracts/Contest.sol";
import "../contracts/interfaces/IContestFactory.sol";
import "../contracts/interfaces/IContest.sol";
import "../contracts/InfToken.sol";

contract ContestFactoryTest is Test {
    ContestFactory public factory;
    Contest public contestTemplate;
    InfToken public infToken;
    
    address public owner = address(0x1);
    address public creator1 = address(0x2);
    address public creator2 = address(0x3);
    address public validator1 = address(0x4);
    address public validator2 = address(0x5);
    
    string constant METADATA_URI = "ipfs://test-metadata";
    uint256 constant DURATION = 7 days;
    bytes32 constant SCORING_MODEL_HASH = bytes32(uint256(0x123456789));
    
    event ContestCreated(
        address indexed contestAddress, 
        address indexed creator, 
        string metadataURI, 
        uint256 duration,
        bytes32 scoringModelHash,
        address[] validators
    );
    event ContestTemplateUpdated(address indexed newTemplate);
    
    function setUp() public {
        contestTemplate = new Contest();
        infToken = new InfToken();
        
        vm.prank(owner);
        factory = new ContestFactory(address(contestTemplate), address(infToken));
    }
    
    function testConstructor() public {
        assertEq(factory.contestTemplate(), address(contestTemplate));
        assertEq(factory.infToken(), address(infToken));
        assertEq(factory.owner(), owner);
        assertEq(factory.getTotalContestsCreated(), 0);
    }
    
    function testConstructorRevert() public {
        vm.expectRevert("Invalid template");
        new ContestFactory(address(0), address(infToken));
        
        vm.expectRevert("Invalid InfToken");
        new ContestFactory(address(contestTemplate), address(0));
    }
    
    function testCreateContest() public {
        address[] memory validators = new address[](2);
        validators[0] = validator1;
        validators[1] = validator2;
        
        IContestFactory.ContestConfig memory config = IContestFactory.ContestConfig({
            metadataURI: METADATA_URI,
            duration: DURATION,
            scoringModelHash: SCORING_MODEL_HASH,
            validators: validators
        });
        
        vm.expectEmit(false, true, false, true);
        emit ContestCreated(address(0), creator1, METADATA_URI, DURATION, SCORING_MODEL_HASH, validators);
        
        vm.prank(creator1);
        address contestAddress = factory.createContest(config);
        
        // Verify contest was created correctly
        assertTrue(contestAddress != address(0));
        assertTrue(factory.isValidContest(contestAddress));
        
        // Check contest initialization
        IContest contest = IContest(contestAddress);
        IContest.ContestInfo memory info = contest.getContestInfo();
        
        assertEq(info.creator, creator1);
        assertEq(info.metadataURI, METADATA_URI);
        assertEq(info.duration, DURATION);
        assertEq(uint256(info.status), uint256(IContest.ContestStatus.Active));
        assertEq(info.scoringModelHash, SCORING_MODEL_HASH);
        assertEq(info.validators.length, 2);
        assertTrue(info.startTime > 0);
        
        // Check validators
        assertTrue(contest.isValidator(validator1));
        assertTrue(contest.isValidator(validator2));
        
        // Check factory tracking
        assertEq(factory.getTotalContestsCreated(), 1);
        
        address[] memory allContests = factory.getCreatedContests();
        assertEq(allContests.length, 1);
        assertEq(allContests[0], contestAddress);
        
        address[] memory creatorContests = factory.getContestsByCreator(creator1);
        assertEq(creatorContests.length, 1);
        assertEq(creatorContests[0], contestAddress);
    }
    
    function testCreateContestRevert() public {
        address[] memory validators = new address[](1);
        validators[0] = validator1;
        
        IContestFactory.ContestConfig memory config = IContestFactory.ContestConfig({
            metadataURI: "",
            duration: DURATION,
            scoringModelHash: SCORING_MODEL_HASH,
            validators: validators
        });
        
        vm.expectRevert("Empty metadata URI");
        vm.prank(creator1);
        factory.createContest(config);
        
        // Test invalid scoring model hash
        config.metadataURI = METADATA_URI;
        config.scoringModelHash = bytes32(0);
        
        vm.expectRevert("Invalid scoring model hash");
        vm.prank(creator1);
        factory.createContest(config);
        
        // Test empty validators
        config.scoringModelHash = SCORING_MODEL_HASH;
        address[] memory emptyValidators = new address[](0);
        config.validators = emptyValidators;
        
        vm.expectRevert("Must have at least one validator");
        vm.prank(creator1);
        factory.createContest(config);
    }
    
    function testCreateMultipleContests() public {
        address[] memory validators1 = new address[](1);
        validators1[0] = validator1;
        
        address[] memory validators2 = new address[](2);
        validators2[0] = validator1;
        validators2[1] = validator2;
        
        IContestFactory.ContestConfig memory config1 = IContestFactory.ContestConfig({
            metadataURI: "ipfs://contest-1",
            duration: DURATION,
            scoringModelHash: SCORING_MODEL_HASH,
            validators: validators1
        });
        
        IContestFactory.ContestConfig memory config2 = IContestFactory.ContestConfig({
            metadataURI: "ipfs://contest-2",
            duration: DURATION * 2,
            scoringModelHash: bytes32(uint256(0x987654321)),
            validators: validators2
        });
        
        // Create contests from different creators
        vm.prank(creator1);
        address contest1 = factory.createContest(config1);
        
        vm.prank(creator1);
        address contest2 = factory.createContest(config2);
        
        vm.prank(creator2);
        address contest3 = factory.createContest(config1);
        
        // Verify total count
        assertEq(factory.getTotalContestsCreated(), 3);
        
        // Verify all contests
        address[] memory allContests = factory.getCreatedContests();
        assertEq(allContests.length, 3);
        assertEq(allContests[0], contest1);
        assertEq(allContests[1], contest2);
        assertEq(allContests[2], contest3);
        
        // Verify creator1's contests
        address[] memory creator1Contests = factory.getContestsByCreator(creator1);
        assertEq(creator1Contests.length, 2);
        assertEq(creator1Contests[0], contest1);
        assertEq(creator1Contests[1], contest2);
        
        // Verify creator2's contests
        address[] memory creator2Contests = factory.getContestsByCreator(creator2);
        assertEq(creator2Contests.length, 1);
        assertEq(creator2Contests[0], contest3);
        
        // Verify all are valid
        assertTrue(factory.isValidContest(contest1));
        assertTrue(factory.isValidContest(contest2));
        assertTrue(factory.isValidContest(contest3));
        assertFalse(factory.isValidContest(address(0x999)));
    }
    
    function testSetContestTemplate() public {
        Contest newTemplate = new Contest();
        
        vm.expectEmit(true, false, false, false);
        emit ContestTemplateUpdated(address(newTemplate));
        
        vm.prank(owner);
        factory.setContestTemplate(address(newTemplate));
        
        assertEq(factory.getContestTemplate(), address(newTemplate));
    }
    
    function testSetContestTemplateRevert() public {
        // Test non-owner setting template
        Contest newTemplate = new Contest();
        
        vm.expectRevert("Ownable: caller is not the owner");
        vm.prank(creator1);
        factory.setContestTemplate(address(newTemplate));
        
        // Test setting invalid template
        vm.expectRevert("Invalid template");
        vm.prank(owner);
        factory.setContestTemplate(address(0));
    }
    
    function testSetInfToken() public {
        InfToken newInfToken = new InfToken();
        
        vm.prank(owner);
        factory.setInfToken(address(newInfToken));
        
        assertEq(factory.getInfToken(), address(newInfToken));
    }
    
    function testSetInfTokenRevert() public {
        // Test non-owner setting InfToken
        InfToken newInfToken = new InfToken();
        
        vm.expectRevert("Ownable: caller is not the owner");
        vm.prank(creator1);
        factory.setInfToken(address(newInfToken));
        
        // Test setting invalid InfToken
        vm.expectRevert("Invalid InfToken");
        vm.prank(owner);
        factory.setInfToken(address(0));
    }
    
    function testGetContestsByCreatorEmpty() public {
        address[] memory contests = factory.getContestsByCreator(creator1);
        assertEq(contests.length, 0);
    }
    
    function testContestFunctionality() public {
        // Create a contest and test that it works properly
        address[] memory validators = new address[](1);
        validators[0] = validator1;
        
        IContestFactory.ContestConfig memory config = IContestFactory.ContestConfig({
            metadataURI: METADATA_URI,
            duration: DURATION,
            scoringModelHash: SCORING_MODEL_HASH,
            validators: validators
        });
        
        vm.prank(creator1);
        address contestAddress = factory.createContest(config);
        
        IContest contest = IContest(contestAddress);
        
        // Test joining contest
        address participant = address(0x6);
        vm.prank(participant);
        contest.joinContest();
        
        assertTrue(contest.isParticipant(participant));
        
        // Test validator functionality
        assertTrue(contest.isValidator(validator1));
        assertFalse(contest.isValidator(participant));
        
        // Test submitting entry (would need verifier to be set)
        // This is tested in the Contest test file
    }
    
    function testNewTemplateUsedForNewContests() public {
        address[] memory validators = new address[](1);
        validators[0] = validator1;
        
        // Create contest with original template
        IContestFactory.ContestConfig memory config = IContestFactory.ContestConfig({
            metadataURI: METADATA_URI,
            duration: DURATION,
            scoringModelHash: SCORING_MODEL_HASH,
            validators: validators
        });
        
        vm.prank(creator1);
        address contest1 = factory.createContest(config);
        
        // Deploy new template and set it
        Contest newTemplate = new Contest();
        vm.prank(owner);
        factory.setContestTemplate(address(newTemplate));
        
        // Create contest with new template
        vm.prank(creator1);
        address contest2 = factory.createContest(config);
        
        // Both should be valid but different contracts
        assertTrue(factory.isValidContest(contest1));
        assertTrue(factory.isValidContest(contest2));
        assertTrue(contest1 != contest2);
        
        // Both should work the same way
        IContest c1 = IContest(contest1);
        IContest c2 = IContest(contest2);
        
        assertEq(c1.getContestInfo().creator, creator1);
        assertEq(c2.getContestInfo().creator, creator1);
    }
} 