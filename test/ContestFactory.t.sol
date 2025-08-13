// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Test.sol";
import "../contracts/ContestFactory.sol";
import "../contracts/Contest.sol";
import "../contracts/interfaces/IContestFactory.sol";
import "../contracts/interfaces/IContest.sol";
import "../contracts/InfToken.sol";
import "../contracts/ModelRegistry.sol";
import "../contracts/ZKVerifierRegistry.sol";
import "../contracts/MockVerifier.sol";

contract ContestFactoryTest is Test {
    ContestFactory public factory;
    Contest public contestTemplate;
    InfToken public infToken;
    ModelRegistry public modelRegistry;
    ZKVerifierRegistry public verifierRegistry;
    MockVerifier public mockVerifier;
    
    address public owner = address(0x1);
    address public creator1 = address(0x2);
    address public creator2 = address(0x3);
    address public validator1 = address(0x4);
    address public validator2 = address(0x5);
    
    string constant METADATA_URI = "ipfs://test-metadata";
    uint256 constant DURATION = 7 days;
    uint256 constant EPOCH_DURATION = 1 days;
    bytes32 constant SCORING_MODEL_HASH = bytes32(uint256(0x123456789));
    uint256 constant REWARD_AMOUNT = 1000 * 10**18;
    
    event ContestCreated(
        address indexed contestAddress, 
        address indexed creator, 
        string metadataURI, 
        uint256 duration,
        uint256 epochDuration,
        bytes32 scoringModelHash,
        address[] validators,
        uint256 rewardAmount
    );
    event ContestTemplateUpdated(address indexed template);
    
    function setUp() public {
        contestTemplate = new Contest();
        infToken = new InfToken();
        modelRegistry = new ModelRegistry();
        verifierRegistry = new ZKVerifierRegistry();
        mockVerifier = new MockVerifier();
        
        // Register scoring model in verifier registry
        vm.prank(address(this));
        verifierRegistry.registerVerifier(
            SCORING_MODEL_HASH,
            address(mockVerifier),
            "ipfs://scoring-model-metadata"
        );
        
        vm.prank(owner);
        factory = new ContestFactory(
            address(contestTemplate), 
            address(infToken),
            address(modelRegistry),
            address(verifierRegistry)
        );
        
        // Fund creators with tokens
        vm.deal(creator1, 10 ether);
        vm.deal(creator2, 10 ether);
        infToken.transfer(creator1, 2000 * 10**18);
        infToken.transfer(creator2, 2000 * 10**18);
    }
    
    function testConstructor() public {
        assertEq(factory.contestTemplate(), address(contestTemplate));
        assertEq(factory.infToken(), address(infToken));
        assertEq(factory.modelRegistry(), address(modelRegistry));
        assertEq(factory.verifierRegistry(), address(verifierRegistry));
        assertEq(factory.owner(), owner);
        assertEq(factory.getContestCount(), 0);
    }
    
    function testConstructorRevert() public {
        vm.expectRevert("Invalid template");
        new ContestFactory(address(0), address(infToken), address(modelRegistry), address(verifierRegistry));
        
        vm.expectRevert("Invalid InfToken");
        new ContestFactory(address(contestTemplate), address(0), address(modelRegistry), address(verifierRegistry));
        
        vm.expectRevert("Invalid model registry");
        new ContestFactory(address(contestTemplate), address(infToken), address(0), address(verifierRegistry));
        
        vm.expectRevert("Invalid verifier registry");
        new ContestFactory(address(contestTemplate), address(infToken), address(modelRegistry), address(0));
    }
    
    function testCreateContest() public {
        address[] memory validators = new address[](2);
        validators[0] = validator1;
        validators[1] = validator2;
        
        IContestFactory.ContestConfig memory config = IContestFactory.ContestConfig({
            metadataURI: METADATA_URI,
            duration: DURATION,
            epochDuration: EPOCH_DURATION,
            scoringModelHash: SCORING_MODEL_HASH,
            validators: validators,
            rewardAmount: REWARD_AMOUNT
        });
        
        // Approve factory to spend tokens
        vm.prank(creator1);
        infToken.approve(address(factory), REWARD_AMOUNT);
        
        // Don't expect exact event since contest address is unknown
        
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
        assertEq(info.epochDuration, EPOCH_DURATION);
        assertEq(uint256(info.status), uint256(IContest.ContestStatus.Active));
        assertEq(info.scoringModelHash, SCORING_MODEL_HASH);
        assertEq(info.validators.length, 2);
        assertTrue(info.startTime > 0);
        
        // Check validators
        assertTrue(contest.isValidator(validator1));
        assertTrue(contest.isValidator(validator2));
        
        // Check factory tracking
        assertEq(factory.getContestCount(), 1);
        
        address[] memory allContests = factory.getCreatedContests();
        assertEq(allContests.length, 1);
        assertEq(allContests[0], contestAddress);
        
        address[] memory creatorContests = factory.getContestsByCreator(creator1);
        assertEq(creatorContests.length, 1);
        assertEq(creatorContests[0], contestAddress);
        
        // Check that tokens were transferred to contest
        assertEq(infToken.balanceOf(contestAddress), REWARD_AMOUNT);
    }
    
    function testCreateContestRevert() public {
        address[] memory validators = new address[](2);
        validators[0] = validator1;
        validators[1] = validator2;
        
        // Test empty metadata URI
        IContestFactory.ContestConfig memory config = IContestFactory.ContestConfig({
            metadataURI: "",
            duration: DURATION,
            epochDuration: EPOCH_DURATION,
            scoringModelHash: SCORING_MODEL_HASH,
            validators: validators,
            rewardAmount: REWARD_AMOUNT
        });
        
        vm.prank(creator1);
        infToken.approve(address(factory), REWARD_AMOUNT);
        
        vm.prank(creator1);
        vm.expectRevert("Empty metadata URI");
        factory.createContest(config);
        
        // Test invalid scoring model hash
        config.metadataURI = METADATA_URI;
        config.scoringModelHash = bytes32(0);
        
        vm.prank(creator1);
        vm.expectRevert("Invalid scoring model hash");
        factory.createContest(config);
        
        // Test no validators
        config.scoringModelHash = SCORING_MODEL_HASH;
        address[] memory emptyValidators = new address[](0);
        config.validators = emptyValidators;
        
        vm.prank(creator1);
        vm.expectRevert("Must have at least one validator");
        factory.createContest(config);
        
        // Test zero reward amount
        config.validators = validators;
        config.rewardAmount = 0;
        
        vm.prank(creator1);
        vm.expectRevert("Must provide reward amount");
        factory.createContest(config);
        
        // Test inactive scoring model
        config.rewardAmount = REWARD_AMOUNT;
        config.scoringModelHash = bytes32(uint256(0x999));
        
        vm.prank(creator1);
        vm.expectRevert("Scoring model not registered or inactive");
        factory.createContest(config);
    }
    
    function testMultipleContests() public {
        address[] memory validators = new address[](1);
        validators[0] = validator1;
        
        IContestFactory.ContestConfig memory config = IContestFactory.ContestConfig({
            metadataURI: METADATA_URI,
            duration: DURATION,
            epochDuration: EPOCH_DURATION,
            scoringModelHash: SCORING_MODEL_HASH,
            validators: validators,
            rewardAmount: REWARD_AMOUNT
        });
        
        // Create first contest
        vm.prank(creator1);
        infToken.approve(address(factory), REWARD_AMOUNT);
        vm.prank(creator1);
        address contest1 = factory.createContest(config);
        
        // Create second contest
        vm.prank(creator2);
        infToken.approve(address(factory), REWARD_AMOUNT);
        vm.prank(creator2);
        address contest2 = factory.createContest(config);
        
        // Verify both contests were created
        assertEq(factory.getContestCount(), 2);
        assertTrue(factory.isValidContest(contest1));
        assertTrue(factory.isValidContest(contest2));
        
        // Verify creator contests
        address[] memory creator1Contests = factory.getContestsByCreator(creator1);
        assertEq(creator1Contests.length, 1);
        assertEq(creator1Contests[0], contest1);
        
        address[] memory creator2Contests = factory.getContestsByCreator(creator2);
        assertEq(creator2Contests.length, 1);
        assertEq(creator2Contests[0], contest2);
    }
    
    function testSetContestTemplate() public {
        Contest newTemplate = new Contest();
        
        vm.prank(owner);
        factory.setContestTemplate(address(newTemplate));
        
        assertEq(factory.contestTemplate(), address(newTemplate));
    }
    
    function testSetContestTemplateRevert() public {
        vm.prank(creator1);
        vm.expectRevert();
        factory.setContestTemplate(address(0));
    }
    
    function testSetInfToken() public {
        InfToken newToken = new InfToken();
        
        vm.prank(owner);
        factory.setInfToken(address(newToken));
        
        assertEq(factory.infToken(), address(newToken));
    }
    
    function testSetModelRegistry() public {
        ModelRegistry newRegistry = new ModelRegistry();
        
        vm.prank(owner);
        factory.setModelRegistry(address(newRegistry));
        
        assertEq(factory.modelRegistry(), address(newRegistry));
    }
    
    function testSetVerifierRegistry() public {
        ZKVerifierRegistry newRegistry = new ZKVerifierRegistry();
        
        vm.prank(owner);
        factory.setVerifierRegistry(address(newRegistry));
        
        assertEq(factory.verifierRegistry(), address(newRegistry));
    }
} 